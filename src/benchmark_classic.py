"""
ComputeLLM - Module de benchmarks classiques.
Tests de performance CPU (single/multi-thread), GPU et mémoire.
Inclut un moniteur de ressources en temps réel.
"""

import time
import threading
import multiprocessing
import platform
import gc
import json
import re
import subprocess
import sys
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field

import numpy as np
import psutil

from src.config import CLASSIC_BENCHMARK_CONFIG, MONITOR_INTERVAL


# =============================================================================
# Nettoyage mémoire avant benchmark
# =============================================================================

def _memory_pressure_flush(target_fraction: float = 0.5) -> None:
    """
    Force macOS à libérer ses caches fichier / mémoire inactive
    en créant une pression mémoire artificielle.
    On alloue progressivement de gros blocs jusqu'à target_fraction
    de la RAM disponible, puis on libère tout. macOS evicte alors
    les caches pour satisfaire la demande.
    On fait deux passes pour maximiser la libération.
    """
    for _ in range(2):
        vm = psutil.virtual_memory()
        target_bytes = min(
            int(vm.available * target_fraction),
            4 * 1024 * 1024 * 1024,
        )
        block_size = 64 * 1024 * 1024  # 64 Mo par bloc
        blocks = []
        allocated = 0
        try:
            while allocated < target_bytes:
                chunk = bytearray(block_size)
                # Toucher chaque page (4 Ko) pour forcer le commit réel
                for offset in range(0, block_size, 4096):
                    chunk[offset] = 0xFF
                blocks.append(chunk)
                allocated += block_size
        except MemoryError:
            pass
        finally:
            del blocks
            gc.collect()
        # Pause entre les passes pour laisser l'OS stabiliser
        time.sleep(0.5)


def system_cleanup(label: str = "", verbose: bool = True) -> Dict[str, float]:
    """
    Nettoyage agressif de la mémoire avant un benchmark.
    Libère le garbage collector Python, les caches GPU, puis force
    macOS à libérer ses caches fichier via pression mémoire artificielle.
    
    Returns:
        Dict avec la RAM disponible avant/après le nettoyage.
    """
    ram_before = psutil.virtual_memory().available / (1024**3)
    used_before = psutil.virtual_memory().used / (1024**3)

    # 1. Garbage collector Python (3 générations)
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)

    # 2. Purge des caches GPU si PyTorch est chargé
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            try:
                torch.xpu.empty_cache()
                torch.xpu.synchronize()
            except AttributeError:
                pass
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except (ImportError, Exception):
        pass

    # 3. Purge des caches Python internes (modules, linecache, etc.)
    try:
        import linecache
        linecache.clearcache()
    except Exception:
        pass
    try:
        import importlib
        importlib.invalidate_caches()
    except Exception:
        pass

    # 4. Purge des caches OS
    if platform.system() == "Darwin":
        purged = False
        try:
            r = subprocess.run(
                ["sudo", "-n", "purge"],
                capture_output=True, timeout=5,
            )
            purged = r.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass

        if not purged:
            _memory_pressure_flush(target_fraction=0.6)
    elif platform.system() == "Linux":
        try:
            subprocess.run(
                ["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
                capture_output=True, timeout=5,
            )
        except Exception:
            _memory_pressure_flush(target_fraction=0.5)

    # 5. Dernier GC après purge
    gc.collect()

    # Pause pour laisser l'OS stabiliser les compteurs mémoire
    time.sleep(1.0)

    vm = psutil.virtual_memory()
    ram_after = vm.available / (1024**3)
    used_after = vm.used / (1024**3)
    freed = used_before - used_after  # Réduction de la mémoire utilisée

    result = {
        "ram_available_before_gb": round(ram_before, 2),
        "ram_available_after_gb": round(ram_after, 2),
        "ram_used_before_gb": round(used_before, 2),
        "ram_used_after_gb": round(used_after, 2),
        "ram_freed_gb": round(freed, 2),
    }

    if verbose and label:
        print(f"  🧹 Nettoyage ({label}) : {used_before:.2f} → {used_after:.2f} Go utilisés "
              f"({'+' if freed >= 0 else ''}{freed:.2f} Go libérés)")

    return result


# =============================================================================
# Moniteur de ressources (CPU, RAM, GPU) en arrière-plan
# =============================================================================

class ResourceMonitor:
    """
    Moniteur de ressources système exécuté dans un thread séparé.
    Échantillonne CPU, RAM et GPU à intervalles réguliers pendant un benchmark.
    """

    def __init__(self, interval: float = MONITOR_INTERVAL):
        self.interval = interval
        self.running = False
        self.samples: List[Dict[str, Any]] = []
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Démarre le monitoring en arrière-plan."""
        self.running = True
        self.samples = []
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        """Arrête le monitoring et retourne un résumé."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        return self.get_summary()

    def _monitor_loop(self):
        """Boucle principale de monitoring."""
        while self.running:
            try:
                sample = {
                    "timestamp": time.time(),
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "cpu_per_core": psutil.cpu_percent(percpu=True),
                    "ram_used_gb": psutil.virtual_memory().used / (1024**3),
                    "ram_percent": psutil.virtual_memory().percent,
                    "ram_available_gb": psutil.virtual_memory().available / (1024**3),
                }

                # Monitoring GPU NVIDIA si disponible
                gpu_usage = self._sample_nvidia_gpu()
                if gpu_usage:
                    sample.update(gpu_usage)
                else:
                    # Monitoring GPU AMD si disponible
                    amd_usage = self._sample_amd_gpu()
                    if amd_usage:
                        sample.update(amd_usage)
                    else:
                        # Monitoring GPU Intel si disponible
                        intel_usage = self._sample_intel_gpu()
                        if intel_usage:
                            sample.update(intel_usage)

                self.samples.append(sample)
            except Exception:
                pass

            time.sleep(self.interval)

    def _sample_nvidia_gpu(self) -> Optional[Dict[str, Any]]:
        """Échantillonne l'utilisation GPU NVIDIA via nvidia-smi."""
        # Ignorer nvidia-smi si on est sur ROCm (AMD)
        try:
            import torch
            if hasattr(torch.version, "hip") and torch.version.hip:
                return None
        except (ImportError, Exception):
            pass
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                parts = [p.strip() for p in result.stdout.strip().split(',')]
                if len(parts) >= 4:
                    return {
                        "gpu_utilization_percent": float(parts[0]),
                        "gpu_memory_used_mb": float(parts[1]),
                        "gpu_memory_total_mb": float(parts[2]),
                        "gpu_temperature_c": float(parts[3]),
                    }
        except (FileNotFoundError, Exception):
            pass
        return None

    def _sample_amd_gpu(self) -> Optional[Dict[str, Any]]:
        """Échantillonne l'utilisation GPU AMD via rocm-smi."""
        try:
            import subprocess
            result = subprocess.run(
                ["rocm-smi", "--showuse", "--showmemuse", "--showtemp", "--csv"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_data: Dict[str, Any] = {}
                for line in lines:
                    lower = line.lower()
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) < 2:
                        continue
                    val_str = parts[1].strip().rstrip('%')
                    if "gpu use" in lower or "gpu busy" in lower:
                        try:
                            gpu_data["gpu_utilization_percent"] = float(val_str)
                        except ValueError:
                            pass
                    elif "vram use" in lower or "gpu memory use" in lower:
                        try:
                            gpu_data["gpu_memory_used_mb"] = float(val_str)
                        except ValueError:
                            pass
                    elif "temperature" in lower or "temp" in lower:
                        try:
                            gpu_data["gpu_temperature_c"] = float(val_str)
                        except ValueError:
                            pass
                if "gpu_utilization_percent" in gpu_data:
                    # Assurer des valeurs par défaut
                    gpu_data.setdefault("gpu_memory_used_mb", 0.0)
                    gpu_data.setdefault("gpu_temperature_c", 0.0)
                    return gpu_data
        except (FileNotFoundError, Exception):
            pass
        # Fallback: rocm-smi showgpuuse format texte
        try:
            import subprocess
            result = subprocess.run(
                ["rocm-smi", "-u"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                usage = 0.0
                for line in result.stdout.strip().split('\n'):
                    if "gpu use" in line.lower() or "busy" in line.lower():
                        match = re.search(r'(\d+\.?\d*)', line)
                        if match:
                            usage = float(match.group(1))
                if usage > 0:
                    return {
                        "gpu_utilization_percent": usage,
                        "gpu_memory_used_mb": 0.0,
                        "gpu_temperature_c": 0.0,
                    }
        except (FileNotFoundError, Exception):
            pass
        return None

    def _sample_intel_gpu(self) -> Optional[Dict[str, Any]]:
        """Échantillonne l'utilisation GPU Intel via xpu-smi."""
        try:
            import subprocess
            result = subprocess.run(
                ["xpu-smi", "dump", "-d", "0", "-m", "0,5,18", "-n", "1"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in reversed(lines):
                    if line.startswith("Timestamp"):
                        continue
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        try:
                            return {
                                "gpu_utilization_percent": float(parts[1]) if parts[1] != "N/A" else 0.0,
                                "gpu_memory_used_mb": float(parts[2]) if parts[2] != "N/A" else 0.0,
                                "gpu_temperature_c": float(parts[3]) if parts[3] != "N/A" else 0.0,
                            }
                        except (ValueError, IndexError):
                            pass
        except (FileNotFoundError, Exception):
            pass
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Calcule un résumé statistique des échantillons collectés."""
        if not self.samples:
            return {"error": "Aucun échantillon collecté"}

        cpu_values = [s["cpu_percent"] for s in self.samples]
        ram_values = [s["ram_percent"] for s in self.samples]

        summary = {
            "n_samples": len(self.samples),
            "duration_s": self.samples[-1]["timestamp"] - self.samples[0]["timestamp"] if len(self.samples) > 1 else 0,
            "cpu": {
                "avg_percent": round(np.mean(cpu_values), 1),
                "max_percent": round(max(cpu_values), 1),
                "min_percent": round(min(cpu_values), 1),
            },
            "ram": {
                "avg_percent": round(np.mean(ram_values), 1),
                "max_percent": round(max(ram_values), 1),
                "peak_used_gb": round(max(s["ram_used_gb"] for s in self.samples), 2),
            },
        }

        # GPU si présent
        gpu_samples = [s for s in self.samples if "gpu_utilization_percent" in s]
        if gpu_samples:
            gpu_util = [s["gpu_utilization_percent"] for s in gpu_samples]
            gpu_mem = [s["gpu_memory_used_mb"] for s in gpu_samples]
            gpu_temp = [s["gpu_temperature_c"] for s in gpu_samples]
            summary["gpu"] = {
                "avg_utilization_percent": round(np.mean(gpu_util), 1),
                "max_utilization_percent": round(max(gpu_util), 1),
                "peak_memory_mb": round(max(gpu_mem), 1),
                "max_temperature_c": round(max(gpu_temp), 1),
            }

        return summary


# =============================================================================
# Benchmarks CPU
# =============================================================================

def _matrix_multiply_task(size: int) -> float:
    """Effectue une multiplication matricielle et retourne le temps en secondes."""
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)

    start = time.perf_counter()
    _ = np.dot(A, B)
    elapsed = time.perf_counter() - start

    return elapsed


def _run_matmul_subprocess(
    size: int,
    n_iterations: int,
    max_threads: Optional[int] = None,
) -> List[float]:
    """
    Exécute les multiplications matricielles dans un sous-processus isolé.

    Lancer un sous-processus frais garantit :
    - Aucune interférence avec le thread de monitoring ou Streamlit
    - Un environnement BLAS propre (variables d'env appliquées AVANT l'import NumPy)
    - Des résultats reproductibles indépendants de l'état du processus principal

    Args:
        size: Taille de la matrice carrée (ex. 2048 pour 2048x2048).
        n_iterations: Nombre de multiplications chronométrées.
        max_threads: Nombre max de threads BLAS.
                     1 = single-thread, None = défaut système (multi-thread).

    Returns:
        Liste des temps d'exécution (en secondes) pour chaque itération.
    """
    # Construction du bloc de configuration des threads
    if max_threads is not None:
        thread_setup = (
            f'import os\n'
            f'os.environ["OMP_NUM_THREADS"] = "{max_threads}"\n'
            f'os.environ["MKL_NUM_THREADS"] = "{max_threads}"\n'
            f'os.environ["OPENBLAS_NUM_THREADS"] = "{max_threads}"\n'
            f'os.environ["VECLIB_MAXIMUM_THREADS"] = "{max_threads}"\n'
            f'os.environ["NUMEXPR_NUM_THREADS"] = "{max_threads}"\n'
        )
    else:
        thread_setup = ""  # Laisser le défaut système (tous les threads)

    script = f"""
{thread_setup}
import numpy as np
import time
import json

np.random.seed(42)
size = {size}
n_iters = {n_iterations}

# Warmup : stabiliser frequence CPU et caches
A = np.random.rand(size, size).astype(np.float32)
B = np.random.rand(size, size).astype(np.float32)
_ = np.dot(A, B)

times = []
for i in range(n_iters):
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)
    start = time.perf_counter()
    _ = np.dot(A, B)
    elapsed = time.perf_counter() - start
    times.append(elapsed)

print(json.dumps(times))
"""
    label = f"{max_threads}-thread" if max_threads else "multi-thread"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Erreur sous-processus {label}: {result.stderr[:500]}"
        )
    # Parser la dernière ligne (ignorer d'éventuels warnings NumPy)
    output_lines = result.stdout.strip().split('\n')
    return json.loads(output_lines[-1])


def benchmark_cpu_single_thread(
    matrix_sizes: List[int] = None,
    n_iterations: int = None,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Benchmark CPU single-thread via multiplication matricielle.
    Force NumPy à utiliser un seul thread via un sous-processus isolé.
    """
    if matrix_sizes is None:
        matrix_sizes = CLASSIC_BENCHMARK_CONFIG["matrix_sizes"]
    if n_iterations is None:
        n_iterations = CLASSIC_BENCHMARK_CONFIG["n_iterations"]

    results = {}

    for idx, size in enumerate(matrix_sizes):
        if progress_callback:
            progress_callback(
                idx / len(matrix_sizes),
                f"CPU ST - Matrice {size}x{size} (sous-processus)...",
            )

        # Exécution dans un sous-processus isolé avec env vars single-thread
        # + warmup intégré pour stabiliser la fréquence CPU
        times = _run_matmul_subprocess(size, n_iterations, max_threads=1)

        median_time = float(np.median(times))
        gflops = (2 * size**3) / (median_time * 1e9)
        results[f"{size}x{size}"] = {
            "times_s": [round(t, 4) for t in times],
            "mean_s": round(float(np.mean(times)), 4),
            "median_s": round(median_time, 4),
            "std_s": round(float(np.std(times)), 4),
            "gflops": round(gflops, 2),
        }

        if progress_callback:
            progress_callback(
                (idx + 1) / len(matrix_sizes),
                f"CPU ST - {size}x{size} terminé ({gflops:.1f} GFLOPS)",
            )

    return {
        "test": "CPU Single-Thread",
        "method": "Matrix multiplication (NumPy, 1 thread, subprocess isolé)",
        "results": results,
    }


def benchmark_cpu_multi_thread(
    matrix_sizes: List[int] = None,
    n_iterations: int = None,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Benchmark CPU multi-thread via multiplication matricielle.
    Utilise tous les threads disponibles via NumPy (Accelerate/MKL/OpenBLAS).
    Exécuté dans un sous-processus isolé pour éviter l'interférence
    du thread de monitoring et de Streamlit.
    """
    if matrix_sizes is None:
        matrix_sizes = CLASSIC_BENCHMARK_CONFIG["matrix_sizes"]
    if n_iterations is None:
        n_iterations = CLASSIC_BENCHMARK_CONFIG["n_iterations"]

    results = {}

    for idx, size in enumerate(matrix_sizes):
        if progress_callback:
            progress_callback(
                idx / len(matrix_sizes),
                f"CPU MT - Matrice {size}x{size} (sous-processus)...",
            )

        # Exécution dans un sous-processus isolé sans restriction de threads
        # + warmup intégré pour stabiliser la fréquence CPU
        times = _run_matmul_subprocess(size, n_iterations, max_threads=None)

        median_time = float(np.median(times))
        gflops = (2 * size**3) / (median_time * 1e9)
        results[f"{size}x{size}"] = {
            "times_s": [round(t, 4) for t in times],
            "mean_s": round(float(np.mean(times)), 4),
            "median_s": round(median_time, 4),
            "std_s": round(float(np.std(times)), 4),
            "gflops": round(gflops, 2),
        }

        if progress_callback:
            progress_callback(
                (idx + 1) / len(matrix_sizes),
                f"CPU MT - {size}x{size} terminé ({gflops:.1f} GFLOPS)",
            )

    return {
        "test": "CPU Multi-Thread",
        "method": f"Matrix multiplication (NumPy, {psutil.cpu_count()} threads, subprocess isolé)",
        "n_threads": psutil.cpu_count(),
        "results": results,
    }


# =============================================================================
# Benchmark Mémoire
# =============================================================================

def benchmark_memory_bandwidth(
    size_mb: int = None,
    n_iterations: int = None,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Benchmark de bande passante mémoire.
    Mesure les vitesses de lecture, écriture et copie en mémoire.
    """
    if size_mb is None:
        size_mb = CLASSIC_BENCHMARK_CONFIG["memory_test_size_mb"]
    if n_iterations is None:
        n_iterations = CLASSIC_BENCHMARK_CONFIG["n_iterations"]

    n_elements = (size_mb * 1024 * 1024) // 8  # float64 = 8 bytes
    data_size_gb = (n_elements * 8) / (1024**3)

    results = {}
    total_ops = 3 * (n_iterations + 1)  # 3 tests × (warmup + n_iterations)
    current_step = 0

    # Test d'écriture
    # Warmup : 1 allocation pour initialiser les pages mémoire et TLB
    _w = np.ones(n_elements, dtype=np.float64)
    del _w
    current_step += 1
    if progress_callback:
        progress_callback(current_step / total_ops, "Mémoire - Warmup écriture")

    write_times = []
    for i in range(n_iterations):
        start = time.perf_counter()
        a = np.ones(n_elements, dtype=np.float64)
        elapsed = time.perf_counter() - start
        write_times.append(elapsed)
        current_step += 1
        if progress_callback:
            progress_callback(current_step / total_ops, f"Mémoire - Écriture ({i+1}/{n_iterations})")

    median_write = float(np.median(write_times))
    results["write"] = {
        "mean_s": round(float(np.mean(write_times)), 4),
        "median_s": round(median_write, 4),
        "bandwidth_gb_s": round(data_size_gb / median_write, 2),
    }

    # Test de lecture
    a = np.ones(n_elements, dtype=np.float64)
    # Warmup : 1 lecture pour initialiser les caches
    _ = np.sum(a)
    current_step += 1
    if progress_callback:
        progress_callback(current_step / total_ops, "Mémoire - Warmup lecture")

    read_times = []
    for i in range(n_iterations):
        start = time.perf_counter()
        _ = np.sum(a)
        elapsed = time.perf_counter() - start
        read_times.append(elapsed)
        current_step += 1
        if progress_callback:
            progress_callback(current_step / total_ops, f"Mémoire - Lecture ({i+1}/{n_iterations})")

    median_read = float(np.median(read_times))
    results["read"] = {
        "mean_s": round(float(np.mean(read_times)), 4),
        "median_s": round(median_read, 4),
        "bandwidth_gb_s": round(data_size_gb / median_read, 2),
    }

    # Test de copie
    # Warmup : 1 copie pour initialiser les chemins mémoire
    _c = a.copy()
    del _c
    current_step += 1
    if progress_callback:
        progress_callback(current_step / total_ops, "Mémoire - Warmup copie")

    copy_times = []
    for i in range(n_iterations):
        start = time.perf_counter()
        b = a.copy()
        elapsed = time.perf_counter() - start
        copy_times.append(elapsed)
        current_step += 1
        if progress_callback:
            progress_callback(current_step / total_ops, f"Mémoire - Copie ({i+1}/{n_iterations})")

    median_copy = float(np.median(copy_times))
    results["copy"] = {
        "mean_s": round(float(np.mean(copy_times)), 4),
        "median_s": round(median_copy, 4),
        "bandwidth_gb_s": round(data_size_gb / median_copy, 2),
    }

    del a, b

    return {
        "test": "Memory Bandwidth",
        "method": f"NumPy array operations ({size_mb} Mo)",
        "data_size_gb": round(data_size_gb, 3),
        "results": results,
    }


# =============================================================================
# Benchmark GPU (via PyTorch si disponible)
# =============================================================================

def _detect_gpu_device(
    selected_gpu: dict = None,
    test_label: str = "GPU",
) -> Dict[str, Any]:
    """
    Détecte le device GPU PyTorch à utiliser.

    Retourne un dict avec :
      - "device": torch.device ou None
      - "device_name": str
      - "backend": str
      - "dev_idx": int
      - "skip_result": dict ou None (si GPU non trouvé, contient le résultat à renvoyer)

    Ordre de priorité : CUDA/ROCm → XPU/IPEX → DirectML → MPS
    """
    try:
        import torch
    except ImportError:
        return {
            "device": None,
            "device_name": "",
            "backend": "",
            "dev_idx": 0,
            "skip_result": {
                "test": test_label,
                "status": "skipped",
                "reason": "PyTorch non installé",
            },
        }

    device = None
    device_name = ""
    backend = ""
    dev_idx = selected_gpu.get("device_index", 0) if selected_gpu else 0
    selected_backend = selected_gpu.get("backend", "") if selected_gpu else ""

    if torch.cuda.is_available():
        is_rocm = hasattr(torch.version, "hip") and torch.version.hip

        # Si l'utilisateur a sélectionné un GPU NVIDIA (cuda) mais que PyTorch
        # est compilé avec ROCm (HIP), CUDA expose en réalité le GPU AMD.
        # On ne doit PAS utiliser ce device pour le benchmark NVIDIA.
        if selected_backend == "cuda" and is_rocm:
            # Ne pas utiliser le device ROCm/HIP pour un benchmark NVIDIA
            device = None
        # Si l'utilisateur a sélectionné un GPU AMD (rocm) mais que PyTorch
        # est compilé avec CUDA (NVIDIA), le device n'est pas le bon non plus.
        elif selected_backend == "rocm" and not is_rocm:
            device = None
        else:
            if dev_idx >= torch.cuda.device_count():
                dev_idx = 0
            device = torch.device(f"cuda:{dev_idx}")
            device_name = torch.cuda.get_device_name(dev_idx)
            if is_rocm:
                backend = f"ROCm/HIP {torch.version.hip}"
            else:
                backend = "CUDA"
    else:
        try:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                xpu_idx = dev_idx if dev_idx < torch.xpu.device_count() else 0
                device = torch.device(f"xpu:{xpu_idx}")
                dev_idx = xpu_idx
                try:
                    device_name = torch.xpu.get_device_name(xpu_idx)
                except Exception:
                    device_name = "Intel GPU (XPU)"
                backend = "SYCL/XPU"
        except Exception:
            pass

        if device is None:
            try:
                import intel_extension_for_pytorch as ipex  # noqa: F401
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    xpu_idx = dev_idx if dev_idx < torch.xpu.device_count() else 0
                    device = torch.device(f"xpu:{xpu_idx}")
                    dev_idx = xpu_idx
                    try:
                        device_name = torch.xpu.get_device_name(xpu_idx)
                    except Exception:
                        device_name = "Intel GPU (XPU via IPEX)"
                    backend = "SYCL/XPU (IPEX)"
            except Exception:
                pass

    if device is None:
        try:
            import torch_directml
            if torch_directml.is_available():
                dml_count = torch_directml.device_count()
                dml_idx = dev_idx if dev_idx < dml_count else 0
                device = torch_directml.device(dml_idx)
                dev_idx = dml_idx
                device_name = torch_directml.device_name(dml_idx)
                backend = "DirectML"
        except ImportError:
            pass

    if device is None:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            device_name = "Apple Silicon (MPS)"
            backend = "Metal/MPS"

    if device is None:
        reason = "Aucun GPU compatible (CUDA/ROCm/XPU/DirectML/MPS) détecté"
        advice = ""
        torch_version = getattr(torch, "__version__", "")
        is_windows = platform.system() == "Windows"

        # ── Priorité : adapter le message au GPU SÉLECTIONNÉ par l'utilisateur ──
        # Si l'utilisateur a explicitement choisi un GPU (ex. NVIDIA), le message
        # d'erreur doit concerner ce GPU-là, pas un autre GPU détecté sur le système.
        selected_backend = selected_gpu.get("backend", "") if selected_gpu else ""
        selected_name = selected_gpu.get("name", "GPU") if selected_gpu else ""

        if selected_backend == "cuda":
            # L'utilisateur a sélectionné un GPU NVIDIA mais CUDA n'est pas dispo
            reason = (
                f"GPU NVIDIA sélectionné ({selected_name}), mais PyTorch n'a pas "
                f"le support CUDA activé (version installée : {torch_version})."
            )
            advice = (
                "Installez PyTorch avec le support CUDA pour utiliser votre GPU NVIDIA :\n"
                "pip install torch torchvision torchaudio "
                "--index-url https://download.pytorch.org/whl/cu124"
            )
        elif selected_backend == "rocm":
            # L'utilisateur a sélectionné un GPU AMD (ROCm)
            reason = (
                f"GPU AMD sélectionné ({selected_name}), mais PyTorch n'a pas "
                f"le support ROCm/HIP activé (version installée : {torch_version})."
            )
            if is_windows:
                advice = (
                    "ROCm n'est pas disponible sur Windows. "
                    "Installez torch-directml pour activer le support GPU AMD :\n"
                    "pip install torch-directml"
                )
            else:
                advice = (
                    "Installez PyTorch avec le support ROCm pour utiliser votre GPU AMD :\n"
                    "pip install torch torchvision torchaudio "
                    "--index-url https://download.pytorch.org/whl/rocm6.2"
                )
        elif selected_backend == "directml":
            reason = (
                f"GPU sélectionné ({selected_name}) utilise DirectML, mais "
                f"torch-directml n'est pas installé."
            )
            advice = (
                "Installez torch-directml pour activer le support GPU via DirectML :\n"
                "pip install torch-directml"
            )
        elif selected_backend == "sycl":
            reason = (
                f"GPU Intel sélectionné ({selected_name}), mais PyTorch n'a pas "
                f"le support XPU/SYCL activé (version installée : {torch_version})."
            )
            advice = (
                "Installez intel-extension-for-pytorch pour activer "
                "le support GPU Intel :\n"
                "pip install intel-extension-for-pytorch"
            )
        elif selected_backend == "metal":
            reason = (
                f"GPU sélectionné ({selected_name}) utilise Metal/MPS, mais "
                f"MPS n'est pas disponible dans votre installation PyTorch."
            )
            advice = (
                "Mettez à jour PyTorch pour activer le support Metal/MPS :\n"
                "pip install --upgrade torch"
            )

        # ── Fallback : auto-détection si aucun GPU n'a été sélectionné ──
        if not advice:
            try:
                from src.hardware_detect import _detect_amd_gpu
                amd_gpus = _detect_amd_gpu()
                if amd_gpus:
                    gpu_name = amd_gpus[0].get("name", "AMD GPU")
                    if "+cu" in torch_version or "cuda" in torch_version.lower():
                        reason = (
                            f"GPU AMD ({gpu_name}) détecté, mais votre PyTorch "
                            f"({torch_version}) est compilé pour CUDA (NVIDIA)."
                        )
                        if is_windows:
                            advice = (
                                "Sur Windows, installez torch-directml pour "
                                "benchmarker votre GPU AMD :\n"
                                "pip install torch-directml"
                            )
                        else:
                            advice = (
                                "Pour benchmarker votre GPU AMD, installez PyTorch "
                                "avec le support ROCm :\n"
                                "pip install torch torchvision torchaudio "
                                "--index-url https://download.pytorch.org/whl/rocm6.2"
                            )
                    else:
                        reason = (
                            f"GPU AMD ({gpu_name}) détecté, mais PyTorch n'a pas "
                            f"le support ROCm activé."
                        )
                        if is_windows:
                            advice = (
                                "ROCm n'est pas disponible sur Windows. "
                                "Installez torch-directml pour activer le support GPU AMD :\n"
                                "pip install torch-directml"
                            )
                        else:
                            advice = (
                                "Installez PyTorch avec le support ROCm pour activer "
                                "le support GPU AMD :\n"
                                "pip install torch torchvision torchaudio "
                                "--index-url https://download.pytorch.org/whl/rocm6.2"
                            )
            except Exception:
                pass

        if not advice:
            try:
                from src.hardware_detect import _detect_intel_gpu
                intel_gpus = _detect_intel_gpu()
                if intel_gpus:
                    gpu_name = intel_gpus[0].get("name", "Intel GPU")
                    if "+cu" in torch_version or "cuda" in torch_version.lower():
                        reason = (
                            f"GPU Intel ({gpu_name}) détecté, mais votre PyTorch "
                            f"({torch_version}) est compilé pour CUDA (NVIDIA)."
                        )
                        advice = (
                            "Pour benchmarker votre GPU Intel, installez PyTorch "
                            "avec le support XPU et intel-extension-for-pytorch :\n"
                            "pip install intel-extension-for-pytorch"
                        )
                    else:
                        reason = (
                            f"GPU Intel ({gpu_name}) détecté, mais PyTorch n'a pas "
                            f"le support XPU activé."
                        )
                        advice = (
                            "Installez intel-extension-for-pytorch pour activer "
                            "le support GPU Intel :\n"
                            "pip install intel-extension-for-pytorch"
                        )
            except Exception:
                pass

        # ── Dernier fallback : si on a trouvé un NVIDIA via nvidia-smi
        # mais torch.cuda.is_available() est False ──
        if not advice:
            try:
                from src.hardware_detect import _detect_nvidia_gpu
                nvidia_gpus = _detect_nvidia_gpu()
                if nvidia_gpus:
                    gpu_name = nvidia_gpus[0].get("name", "NVIDIA GPU")
                    reason = (
                        f"GPU NVIDIA ({gpu_name}) détecté, mais PyTorch n'a pas "
                        f"le support CUDA activé (version installée : {torch_version})."
                    )
                    advice = (
                        "Installez PyTorch avec le support CUDA pour utiliser votre GPU NVIDIA :\n"
                        "pip install torch torchvision torchaudio "
                        "--index-url https://download.pytorch.org/whl/cu124"
                    )
            except Exception:
                pass

        skip = {
            "test": test_label,
            "status": "skipped",
            "reason": reason,
        }
        if advice:
            skip["advice"] = advice

        return {
            "device": None,
            "device_name": "",
            "backend": "",
            "dev_idx": dev_idx,
            "skip_result": skip,
        }

    return {
        "device": device,
        "device_name": device_name,
        "backend": backend,
        "dev_idx": dev_idx,
        "skip_result": None,
    }


def _gpu_sync(device, C=None):
    """Synchronise le GPU quel que soit le backend. Attend la fin du kernel."""
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
        except AttributeError:
            if C is not None:
                C.cpu()
    elif device.type == "privateuseone":
        if C is not None:
            C[0, 0].item()


def _gpu_pre_sync(device, tensor):
    """Barrière pré-sync : attendre que les allocations GPU soient terminées."""
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
        except AttributeError:
            pass
    elif device.type == "privateuseone":
        tensor[0, 0].item()


def _gpu_cleanup(device):
    """Nettoie la mémoire GPU après un benchmark."""
    import torch
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        try:
            torch.xpu.empty_cache()
        except AttributeError:
            pass
    elif device.type == "mps":
        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass


def benchmark_gpu(
    progress_callback: Optional[Callable] = None,
    selected_gpu: dict = None,
) -> Dict[str, Any]:
    """
    Benchmark GPU Raw Compute : multiplication matricielle pure.
    Mesure uniquement la puissance de calcul du GPU (matmul + sync).
    Aucun transfert CPU↔GPU n'est inclus dans le chronométrage.

    Args:
        progress_callback: Callback(progress, message).
        selected_gpu: (Optionnel) Dict GPU avec 'backend' et 'device_index'.
                     Si fourni, utilise ce GPU spécifique.
    """
    gpu_info = _detect_gpu_device(selected_gpu, test_label="GPU Compute (Raw)")
    if gpu_info["skip_result"]:
        return gpu_info["skip_result"]

    import torch

    device = gpu_info["device"]
    device_name = gpu_info["device_name"]
    backend = gpu_info["backend"]
    dev_idx = gpu_info["dev_idx"]

    try:
        sizes = [1024, 2048, 4096]
        results = {}
        total_steps = len(sizes) * 3
        current_step = 0

        for size in sizes:
            times = []

            # Warmup
            A = torch.randn(size, size, device=device, dtype=torch.float32)
            B = torch.randn(size, size, device=device, dtype=torch.float32)
            C = torch.mm(A, B)
            _gpu_sync(device, C)

            for i in range(3):
                A = torch.randn(size, size, device=device, dtype=torch.float32)
                B = torch.randn(size, size, device=device, dtype=torch.float32)

                _gpu_pre_sync(device, A)

                start = time.perf_counter()
                C = torch.mm(A, B)
                _gpu_sync(device, C)
                elapsed = time.perf_counter() - start

                times.append(elapsed)
                current_step += 1

                if progress_callback:
                    progress_callback(current_step / total_steps, f"GPU Raw - Matrice {size}x{size} ({i+1}/3)")

            median_time = float(np.median(times))
            gflops = (2 * size**3) / (median_time * 1e9)
            results[f"{size}x{size}"] = {
                "times_s": [round(t, 4) for t in times],
                "mean_s": round(float(np.mean(times)), 4),
                "median_s": round(median_time, 4),
                "gflops": round(gflops, 2),
            }

            del A, B

        _gpu_cleanup(device)

        return {
            "test": "GPU Compute (Raw)",
            "status": "completed",
            "device": device_name,
            "backend": backend,
            "gpu_index": dev_idx,
            "results": results,
        }

    except Exception as e:
        error_msg = str(e)
        if "level_zero" in error_msg.lower() or "ur_result_error" in error_msg.lower():
            reason = (
                f"GPU {device_name} ({backend}) détecté, mais le driver Level Zero "
                f"a échoué lors de l'exécution. Ce GPU intégré n'est peut-être pas "
                f"entièrement compatible avec les opérations de calcul PyTorch XPU."
            )
            advice = (
                "Les GPU Intel intégrés (Iris, UHD) ont un support XPU limité. "
                "Seuls les GPU Intel Arc et Data Center ont un support complet. "
                "Le benchmark IA via llama-server fonctionne indépendamment."
            )
        else:
            reason = f"Erreur lors du benchmark GPU ({backend}): {error_msg}"
            advice = ""

        result = {
            "test": "GPU Compute (Raw)",
            "status": "skipped",
            "reason": reason,
        }
        if advice:
            result["advice"] = advice
        return result


def benchmark_gpu_system(
    progress_callback: Optional[Callable] = None,
    selected_gpu: dict = None,
) -> Dict[str, Any]:
    """
    Benchmark GPU System Score : pipeline end-to-end.
    Mesure la performance globale GPU incluant les transferts de données :
      1. Allocation CPU → transfert GPU
      2. Calcul GPU (matmul)
      3. Transfert GPU → CPU
    Ce test révèle les différences d'architecture mémoire :
    - Mémoire unifiée (Apple Silicon) : transferts quasi-gratuits
    - GPU discret (NVIDIA/AMD) : coût du bus PCIe
    - DirectML : overhead du runtime DirectX 12

    Args:
        progress_callback: Callback(progress, message).
        selected_gpu: (Optionnel) Dict GPU avec 'backend' et 'device_index'.
    """
    gpu_info = _detect_gpu_device(selected_gpu, test_label="GPU System Score")
    if gpu_info["skip_result"]:
        return gpu_info["skip_result"]

    import torch

    device = gpu_info["device"]
    device_name = gpu_info["device_name"]
    backend = gpu_info["backend"]
    dev_idx = gpu_info["dev_idx"]

    try:
        sizes = [1024, 2048, 4096]
        results = {}
        total_steps = len(sizes) * 3
        current_step = 0

        for size in sizes:
            # Warmup complet du pipeline
            A_cpu = torch.randn(size, size, dtype=torch.float32)
            B_cpu = torch.randn(size, size, dtype=torch.float32)
            A_gpu = A_cpu.to(device)
            B_gpu = B_cpu.to(device)
            C_gpu = torch.mm(A_gpu, B_gpu)
            _ = C_gpu.cpu()
            _gpu_sync(device)
            del A_gpu, B_gpu, C_gpu

            pipeline_times = []     # Temps total end-to-end
            transfer_to_times = []  # CPU → GPU
            compute_times = []      # Matmul pur
            transfer_back_times = []  # GPU → CPU

            for i in range(3):
                # Préparer les données sur CPU
                A_cpu = torch.randn(size, size, dtype=torch.float32)
                B_cpu = torch.randn(size, size, dtype=torch.float32)

                # ── Début pipeline end-to-end ──
                start_total = time.perf_counter()

                # Étape 1 : CPU → GPU
                start_transfer = time.perf_counter()
                A_gpu = A_cpu.to(device)
                B_gpu = B_cpu.to(device)
                _gpu_pre_sync(device, A_gpu)
                transfer_to = time.perf_counter() - start_transfer

                # Étape 2 : Calcul GPU
                start_compute = time.perf_counter()
                C_gpu = torch.mm(A_gpu, B_gpu)
                _gpu_sync(device, C_gpu)
                compute = time.perf_counter() - start_compute

                # Étape 3 : GPU → CPU
                start_back = time.perf_counter()
                C_cpu = C_gpu.cpu()
                transfer_back = time.perf_counter() - start_back

                total = time.perf_counter() - start_total
                # ── Fin pipeline ──

                pipeline_times.append(total)
                transfer_to_times.append(transfer_to)
                compute_times.append(compute)
                transfer_back_times.append(transfer_back)

                del A_gpu, B_gpu, C_gpu, C_cpu
                current_step += 1

                if progress_callback:
                    progress_callback(
                        current_step / total_steps,
                        f"GPU System - Matrice {size}x{size} ({i+1}/3)",
                    )

            # Calcul des métriques (temps médian)
            median_pipeline = float(np.median(pipeline_times))
            median_compute = float(np.median(compute_times))
            median_to = float(np.median(transfer_to_times))
            median_back = float(np.median(transfer_back_times))

            # GFLOPS sur le pipeline total
            gflops_pipeline = (2 * size**3) / (median_pipeline * 1e9)
            # GFLOPS calcul pur (pour comparaison avec Raw)
            gflops_compute = (2 * size**3) / (median_compute * 1e9)

            # Taille des données transférées (2 matrices IN + 1 OUT, float32)
            data_bytes = (2 + 1) * size * size * 4
            data_gb = data_bytes / (1024**3)
            # Bande passante effective du transfert total
            total_transfer_time = median_to + median_back
            transfer_bandwidth_gb_s = data_gb / total_transfer_time if total_transfer_time > 0 else 0

            results[f"{size}x{size}"] = {
                # Temps détaillés
                "pipeline_times_s": [round(t, 4) for t in pipeline_times],
                "pipeline_median_s": round(median_pipeline, 4),
                "transfer_to_median_s": round(median_to, 4),
                "compute_median_s": round(median_compute, 4),
                "transfer_back_median_s": round(median_back, 4),
                # GFLOPS
                "gflops_pipeline": round(gflops_pipeline, 2),
                "gflops_compute": round(gflops_compute, 2),
                # Bande passante transfert
                "transfer_bandwidth_gb_s": round(transfer_bandwidth_gb_s, 2),
                "data_transferred_gb": round(data_gb, 4),
                # Répartition du temps (%)
                "pct_transfer_to": round(100 * median_to / median_pipeline, 1) if median_pipeline > 0 else 0,
                "pct_compute": round(100 * median_compute / median_pipeline, 1) if median_pipeline > 0 else 0,
                "pct_transfer_back": round(100 * median_back / median_pipeline, 1) if median_pipeline > 0 else 0,
            }

        _gpu_cleanup(device)

        return {
            "test": "GPU System Score",
            "status": "completed",
            "device": device_name,
            "backend": backend,
            "gpu_index": dev_idx,
            "results": results,
        }

    except Exception as e:
        error_msg = str(e)
        if "level_zero" in error_msg.lower() or "ur_result_error" in error_msg.lower():
            reason = (
                f"GPU {device_name} ({backend}) détecté, mais le driver Level Zero "
                f"a échoué lors de l'exécution."
            )
            advice = (
                "Les GPU Intel intégrés (Iris, UHD) ont un support XPU limité. "
                "Seuls les GPU Intel Arc et Data Center ont un support complet."
            )
        else:
            reason = f"Erreur lors du benchmark GPU System ({backend}): {error_msg}"
            advice = ""

        result = {
            "test": "GPU System Score",
            "status": "skipped",
            "reason": reason,
        }
        if advice:
            result["advice"] = advice
        return result


# =============================================================================
# Orchestrateur des benchmarks classiques
# =============================================================================

def run_all_classic_benchmarks(
    progress_callback: Optional[Callable] = None,
    selected_gpu: dict = None,
) -> Dict[str, Any]:
    """
    Exécute tous les benchmarks classiques et retourne les résultats consolidés.
    
    Args:
        progress_callback: Fonction callback(progress: float, message: str)
                          progress entre 0.0 et 1.0
        selected_gpu: (Optionnel) Dict GPU avec 'backend' et 'device_index'.
                     Si fourni, utilise ce GPU spécifique pour le benchmark GPU.
    """
    all_results = {}

    # Démarrer le monitoring
    monitor = ResourceMonitor()
    monitor.start()

    start_time = time.time()

    def sub_progress(phase_start, phase_weight):
        """Créée un callback de sous-progression."""
        def callback(p, msg):
            if progress_callback:
                overall = phase_start + (p * phase_weight)
                progress_callback(overall, msg)
        return callback

    try:
        # Phase 1: CPU Single-Thread (0% - 25%)
        try:
            if progress_callback:
                progress_callback(0.0, "Nettoyage mémoire avant CPU Single-Thread...")
            system_cleanup("CPU Single-Thread", verbose=False)
            if progress_callback:
                progress_callback(0.0, "Démarrage benchmark CPU Single-Thread...")
            all_results["cpu_single_thread"] = benchmark_cpu_single_thread(
                progress_callback=sub_progress(0.0, 0.25)
            )
        except Exception as e:
            all_results["cpu_single_thread"] = {
                "test": "CPU Single-Thread", "status": "error",
                "error": str(e),
            }

        # Phase 2: CPU Multi-Thread (25% - 50%)
        try:
            if progress_callback:
                progress_callback(0.25, "Nettoyage mémoire avant CPU Multi-Thread...")
            system_cleanup("CPU Multi-Thread", verbose=False)
            if progress_callback:
                progress_callback(0.25, "Démarrage benchmark CPU Multi-Thread...")
            all_results["cpu_multi_thread"] = benchmark_cpu_multi_thread(
                progress_callback=sub_progress(0.25, 0.25)
            )
        except Exception as e:
            all_results["cpu_multi_thread"] = {
                "test": "CPU Multi-Thread", "status": "error",
                "error": str(e),
            }

        # Phase 3: Mémoire (50% - 65%)
        try:
            if progress_callback:
                progress_callback(0.50, "Nettoyage mémoire avant benchmark Mémoire...")
            system_cleanup("Mémoire", verbose=False)
            if progress_callback:
                progress_callback(0.50, "Démarrage benchmark Mémoire...")
            all_results["memory_bandwidth"] = benchmark_memory_bandwidth(
                progress_callback=sub_progress(0.50, 0.15)
            )
        except Exception as e:
            all_results["memory_bandwidth"] = {
                "test": "Memory Bandwidth", "status": "error",
                "error": str(e),
            }

        # Phase 4: GPU Raw Compute (65% - 82%)
        try:
            if progress_callback:
                progress_callback(0.65, "Nettoyage mémoire avant GPU Raw Compute...")
            system_cleanup("GPU Raw Compute", verbose=False)
            if progress_callback:
                progress_callback(0.65, "Démarrage benchmark GPU Raw Compute...")
            all_results["gpu_compute"] = benchmark_gpu(
                progress_callback=sub_progress(0.65, 0.17),
                selected_gpu=selected_gpu,
            )
        except Exception as e:
            all_results["gpu_compute"] = {
                "test": "GPU Compute (Raw)", "status": "error",
                "error": str(e),
            }

        # Phase 5: GPU System Score (82% - 100%)
        try:
            if progress_callback:
                progress_callback(0.82, "Nettoyage mémoire avant GPU System Score...")
            system_cleanup("GPU System Score", verbose=False)
            if progress_callback:
                progress_callback(0.82, "Démarrage benchmark GPU System Score...")
            all_results["gpu_system"] = benchmark_gpu_system(
                progress_callback=sub_progress(0.82, 0.18),
                selected_gpu=selected_gpu,
            )
        except Exception as e:
            all_results["gpu_system"] = {
                "test": "GPU System Score", "status": "error",
                "error": str(e),
            }

    finally:
        # Arrêter le monitoring
        resource_usage = monitor.stop()

    elapsed_total = time.time() - start_time

    return {
        "type": "classic_benchmarks",
        "total_time_s": round(elapsed_total, 2),
        "benchmarks": all_results,
        "resource_usage": resource_usage,
    }


if __name__ == "__main__":
    def print_progress(p, msg):
        bar = "█" * int(p * 30) + "░" * (30 - int(p * 30))
        print(f"\r  [{bar}] {p*100:.0f}% - {msg}", end="", flush=True)

    print("Lancement des benchmarks classiques...\n")
    results = run_all_classic_benchmarks(progress_callback=print_progress)
    print("\n\nTerminé !")
    
    import json
    print(json.dumps(results, indent=2))
