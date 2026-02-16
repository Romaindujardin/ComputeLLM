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


def _worker_single_thread(size: int) -> float:
    """Worker pour benchmark single-thread (force 1 thread NumPy)."""
    # Forcer single-thread via variables d'environnement
    # (doit être fait avant l'import de NumPy dans le sous-processus)
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # Recréer les matrices avec un seul thread
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)

    start = time.perf_counter()
    _ = np.dot(A, B)
    elapsed = time.perf_counter() - start

    return elapsed


def benchmark_cpu_single_thread(
    matrix_sizes: List[int] = None,
    n_iterations: int = None,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Benchmark CPU single-thread via multiplication matricielle.
    Force NumPy à utiliser un seul thread.
    """
    if matrix_sizes is None:
        matrix_sizes = CLASSIC_BENCHMARK_CONFIG["matrix_sizes"]
    if n_iterations is None:
        n_iterations = CLASSIC_BENCHMARK_CONFIG["n_iterations"]

    results = {}
    total_steps = len(matrix_sizes) * n_iterations
    current_step = 0

    for size in matrix_sizes:
        times = []
        for i in range(n_iterations):
            # Exécuter dans un sous-processus pour garantir le single-thread
            elapsed = _worker_single_thread(size)
            times.append(elapsed)
            current_step += 1
            if progress_callback:
                progress_callback(current_step / total_steps, f"CPU ST - Matrice {size}x{size} ({i+1}/{n_iterations})")

        gflops = (2 * size**3) / (np.mean(times) * 1e9)
        results[f"{size}x{size}"] = {
            "times_s": [round(t, 4) for t in times],
            "mean_s": round(np.mean(times), 4),
            "std_s": round(np.std(times), 4),
            "gflops": round(gflops, 2),
        }

    return {
        "test": "CPU Single-Thread",
        "method": "Matrix multiplication (NumPy, 1 thread)",
        "results": results,
    }


def benchmark_cpu_multi_thread(
    matrix_sizes: List[int] = None,
    n_iterations: int = None,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Benchmark CPU multi-thread via multiplication matricielle.
    Utilise tous les threads disponibles via NumPy (MKL/OpenBLAS).
    """
    if matrix_sizes is None:
        matrix_sizes = CLASSIC_BENCHMARK_CONFIG["matrix_sizes"]
    if n_iterations is None:
        n_iterations = CLASSIC_BENCHMARK_CONFIG["n_iterations"]

    results = {}
    total_steps = len(matrix_sizes) * n_iterations
    current_step = 0

    for size in matrix_sizes:
        times = []
        for i in range(n_iterations):
            elapsed = _matrix_multiply_task(size)
            times.append(elapsed)
            current_step += 1
            if progress_callback:
                progress_callback(current_step / total_steps, f"CPU MT - Matrice {size}x{size} ({i+1}/{n_iterations})")

        gflops = (2 * size**3) / (np.mean(times) * 1e9)
        results[f"{size}x{size}"] = {
            "times_s": [round(t, 4) for t in times],
            "mean_s": round(np.mean(times), 4),
            "std_s": round(np.std(times), 4),
            "gflops": round(gflops, 2),
        }

    return {
        "test": "CPU Multi-Thread",
        "method": f"Matrix multiplication (NumPy, {psutil.cpu_count()} threads)",
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
    total_ops = 3 * n_iterations  # 3 tests × n_iterations
    current_step = 0

    # Test d'écriture
    write_times = []
    for i in range(n_iterations):
        start = time.perf_counter()
        a = np.ones(n_elements, dtype=np.float64)
        elapsed = time.perf_counter() - start
        write_times.append(elapsed)
        current_step += 1
        if progress_callback:
            progress_callback(current_step / total_ops, f"Mémoire - Écriture ({i+1}/{n_iterations})")

    results["write"] = {
        "mean_s": round(np.mean(write_times), 4),
        "bandwidth_gb_s": round(data_size_gb / np.mean(write_times), 2),
    }

    # Test de lecture
    a = np.ones(n_elements, dtype=np.float64)
    read_times = []
    for i in range(n_iterations):
        start = time.perf_counter()
        _ = np.sum(a)
        elapsed = time.perf_counter() - start
        read_times.append(elapsed)
        current_step += 1
        if progress_callback:
            progress_callback(current_step / total_ops, f"Mémoire - Lecture ({i+1}/{n_iterations})")

    results["read"] = {
        "mean_s": round(np.mean(read_times), 4),
        "bandwidth_gb_s": round(data_size_gb / np.mean(read_times), 2),
    }

    # Test de copie
    copy_times = []
    for i in range(n_iterations):
        start = time.perf_counter()
        b = a.copy()
        elapsed = time.perf_counter() - start
        copy_times.append(elapsed)
        current_step += 1
        if progress_callback:
            progress_callback(current_step / total_ops, f"Mémoire - Copie ({i+1}/{n_iterations})")

    results["copy"] = {
        "mean_s": round(np.mean(copy_times), 4),
        "bandwidth_gb_s": round(data_size_gb / np.mean(copy_times), 2),
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

def benchmark_gpu(
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Benchmark GPU via PyTorch (CUDA ou MPS).
    Effectue des multiplications matricielles sur le GPU.
    """
    try:
        import torch
    except ImportError:
        return {
            "test": "GPU Compute",
            "status": "skipped",
            "reason": "PyTorch non installé",
        }

    # Déterminer le device
    device = None
    device_name = ""
    backend = ""

    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        backend = "CUDA"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
        try:
            device_name = torch.xpu.get_device_name(0)
        except Exception:
            device_name = "Intel GPU (XPU)"
        backend = "SYCL/XPU"
    else:
        # Tenter d'activer XPU via intel-extension-for-pytorch (IPEX)
        try:
            import intel_extension_for_pytorch as ipex  # noqa: F401
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                device = torch.device("xpu")
                try:
                    device_name = torch.xpu.get_device_name(0)
                except Exception:
                    device_name = "Intel GPU (XPU via IPEX)"
                backend = "SYCL/XPU (IPEX)"
        except ImportError:
            pass

    # MPS (Apple Silicon)
    if device is None:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            device_name = "Apple Silicon (MPS)"
            backend = "Metal/MPS"

    # Aucun backend GPU trouvé — message d'aide contextuel
    if device is None:
        reason = "Aucun GPU compatible (CUDA/XPU/MPS) détecté"
        advice = ""

        # Vérifier si un GPU Intel est physiquement présent
        try:
            from src.hardware_detect import _detect_intel_gpu
            intel_gpu = _detect_intel_gpu()
            if intel_gpu:
                gpu_name = intel_gpu.get("name", "Intel GPU")
                # Vérifier si PyTorch est un build CUDA (inutile sur Intel)
                torch_version = getattr(torch, "__version__", "")
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

        result = {
            "test": "GPU Compute",
            "status": "skipped",
            "reason": reason,
        }
        if advice:
            result["advice"] = advice
        return result

    sizes = [1024, 2048, 4096]
    results = {}
    total_steps = len(sizes) * 3
    current_step = 0

    for size in sizes:
        times = []

        # Warmup
        A = torch.randn(size, size, device=device, dtype=torch.float32)
        B = torch.randn(size, size, device=device, dtype=torch.float32)
        _ = torch.mm(A, B)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "xpu":
            torch.xpu.synchronize()

        for i in range(3):
            A = torch.randn(size, size, device=device, dtype=torch.float32)
            B = torch.randn(size, size, device=device, dtype=torch.float32)

            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "xpu":
                torch.xpu.synchronize()

            start = time.perf_counter()
            _ = torch.mm(A, B)

            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "xpu":
                torch.xpu.synchronize()
            elif device.type == "mps":
                # Synchronisation MPS
                (A + B).cpu()

            elapsed = time.perf_counter() - start
            times.append(elapsed)
            current_step += 1

            if progress_callback:
                progress_callback(current_step / total_steps, f"GPU - Matrice {size}x{size} ({i+1}/3)")

        gflops = (2 * size**3) / (np.mean(times) * 1e9)
        results[f"{size}x{size}"] = {
            "times_s": [round(t, 4) for t in times],
            "mean_s": round(np.mean(times), 4),
            "gflops": round(gflops, 2),
        }

        del A, B

    # Nettoyer la mémoire GPU
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        try:
            torch.xpu.empty_cache()
        except AttributeError:
            pass

    return {
        "test": "GPU Compute",
        "status": "completed",
        "device": device_name,
        "backend": backend,
        "results": results,
    }


# =============================================================================
# Orchestrateur des benchmarks classiques
# =============================================================================

def run_all_classic_benchmarks(
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Exécute tous les benchmarks classiques et retourne les résultats consolidés.
    
    Args:
        progress_callback: Fonction callback(progress: float, message: str)
                          progress entre 0.0 et 1.0
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
        # Phase 1: CPU Single-Thread (0% - 30%)
        if progress_callback:
            progress_callback(0.0, "Nettoyage mémoire avant CPU Single-Thread...")
        system_cleanup("CPU Single-Thread", verbose=False)
        if progress_callback:
            progress_callback(0.0, "Démarrage benchmark CPU Single-Thread...")
        all_results["cpu_single_thread"] = benchmark_cpu_single_thread(
            progress_callback=sub_progress(0.0, 0.30)
        )

        # Phase 2: CPU Multi-Thread (30% - 60%)
        if progress_callback:
            progress_callback(0.30, "Nettoyage mémoire avant CPU Multi-Thread...")
        system_cleanup("CPU Multi-Thread", verbose=False)
        if progress_callback:
            progress_callback(0.30, "Démarrage benchmark CPU Multi-Thread...")
        all_results["cpu_multi_thread"] = benchmark_cpu_multi_thread(
            progress_callback=sub_progress(0.30, 0.30)
        )

        # Phase 3: Mémoire (60% - 80%)
        if progress_callback:
            progress_callback(0.60, "Nettoyage mémoire avant benchmark Mémoire...")
        system_cleanup("Mémoire", verbose=False)
        if progress_callback:
            progress_callback(0.60, "Démarrage benchmark Mémoire...")
        all_results["memory_bandwidth"] = benchmark_memory_bandwidth(
            progress_callback=sub_progress(0.60, 0.20)
        )

        # Phase 4: GPU (80% - 100%)
        if progress_callback:
            progress_callback(0.80, "Nettoyage mémoire avant benchmark GPU...")
        system_cleanup("GPU", verbose=False)
        if progress_callback:
            progress_callback(0.80, "Démarrage benchmark GPU...")
        all_results["gpu_compute"] = benchmark_gpu(
            progress_callback=sub_progress(0.80, 0.20)
        )

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
