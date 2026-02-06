"""
ComputeLLM - Module de détection matérielle.
Détecte automatiquement le système d'exploitation, CPU, GPU, RAM
et identifie les backends disponibles (CUDA, Metal, CPU).
"""

import platform
import subprocess
import json
import re
import os
import sys
from typing import Dict, Any, Optional


def detect_os() -> Dict[str, str]:
    """Détecte le système d'exploitation et retourne ses informations."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "architecture": platform.architecture()[0],
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }


def detect_cpu() -> Dict[str, Any]:
    """
    Détecte les informations CPU.
    Utilise psutil et des commandes système pour obtenir les détails.
    """
    import psutil

    info: Dict[str, Any] = {
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "architecture": platform.machine(),
    }

    system = platform.system()

    if system == "Darwin":
        info.update(_detect_cpu_macos())
    elif system == "Windows":
        info.update(_detect_cpu_windows())
    elif system == "Linux":
        info.update(_detect_cpu_linux())

    # Fréquence CPU via psutil
    try:
        freq = psutil.cpu_freq()
        if freq:
            info["frequency_mhz"] = {
                "current": round(freq.current, 2),
                "min": round(freq.min, 2) if freq.min else None,
                "max": round(freq.max, 2) if freq.max else None,
            }
    except Exception:
        info["frequency_mhz"] = None

    return info


def _detect_cpu_macos() -> Dict[str, Any]:
    """Détection CPU spécifique macOS."""
    info = {}
    try:
        # Nom du CPU
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["model"] = result.stdout.strip()

        # Vérifier si Apple Silicon
        machine = platform.machine()
        if machine == "arm64":
            info["is_apple_silicon"] = True
            info["architecture_type"] = "ARM (Apple Silicon)"
            # Obtenir le nom du chip
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "hw.chip"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    info["chip"] = result.stdout.strip()
            except Exception:
                pass
            # Performance/efficiency cores
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    info["performance_cores"] = int(result.stdout.strip())
                result = subprocess.run(
                    ["sysctl", "-n", "hw.perflevel1.logicalcpu"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    info["efficiency_cores"] = int(result.stdout.strip())
            except Exception:
                pass
        else:
            info["is_apple_silicon"] = False
            info["architecture_type"] = "x86_64 (Intel)"

    except Exception as e:
        info["detection_error"] = str(e)

    return info


def _detect_cpu_windows() -> Dict[str, Any]:
    """Détection CPU spécifique Windows."""
    info = {}
    try:
        result = subprocess.run(
            ["wmic", "cpu", "get", "Name,MaxClockSpeed,NumberOfCores,NumberOfLogicalProcessors", "/format:csv"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
            if len(lines) >= 2:
                parts = lines[-1].split(',')
                if len(parts) >= 4:
                    info["model"] = parts[2].strip() if len(parts) > 2 else "Unknown"
                    info["max_clock_speed_mhz"] = parts[1].strip() if len(parts) > 1 else None
        info["architecture_type"] = "x86_64" if platform.machine() == "AMD64" else platform.machine()
        info["is_apple_silicon"] = False
    except Exception as e:
        info["detection_error"] = str(e)
        # Fallback
        info["model"] = platform.processor() or "Unknown"
        info["is_apple_silicon"] = False
        info["architecture_type"] = platform.machine()

    return info


def _detect_cpu_linux() -> Dict[str, Any]:
    """Détection CPU spécifique Linux."""
    info = {}
    try:
        with open("/proc/cpuinfo", "r") as f:
            content = f.read()
        match = re.search(r"model name\s*:\s*(.+)", content)
        if match:
            info["model"] = match.group(1).strip()
        info["is_apple_silicon"] = False
        info["architecture_type"] = platform.machine()
    except Exception as e:
        info["detection_error"] = str(e)
        info["model"] = platform.processor() or "Unknown"
    return info


def detect_gpu() -> Dict[str, Any]:
    """
    Détecte le GPU et les backends disponibles.
    Supporte NVIDIA (CUDA), Apple (Metal), et CPU fallback.
    """
    info: Dict[str, Any] = {
        "gpus": [],
        "backends": [],
        "primary_backend": "cpu",
    }

    system = platform.system()

    # --- Détection NVIDIA / CUDA ---
    nvidia_gpu = _detect_nvidia_gpu()
    if nvidia_gpu:
        info["gpus"].append(nvidia_gpu)
        info["backends"].append("cuda")
        info["primary_backend"] = "cuda"

    # --- Détection Metal (macOS) ---
    if system == "Darwin":
        metal_gpu = _detect_metal_gpu()
        if metal_gpu:
            info["gpus"].append(metal_gpu)
            info["backends"].append("metal")
            if info["primary_backend"] == "cpu":
                info["primary_backend"] = "metal"

    # --- CPU fallback toujours disponible ---
    info["backends"].append("cpu")

    # --- Vérification des bibliothèques Python ---
    info["python_backends"] = _detect_python_backends()

    return info


def _detect_nvidia_gpu() -> Optional[Dict[str, Any]]:
    """Détecte un GPU NVIDIA via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
            gpus = []
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    return {
                        "type": "NVIDIA",
                        "name": parts[0],
                        "vram_total_mb": float(parts[1]),
                        "vram_free_mb": float(parts[2]),
                        "driver_version": parts[3],
                        "compute_capability": parts[4],
                        "backend": "cuda",
                    }
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return None


def _detect_metal_gpu() -> Optional[Dict[str, Any]]:
    """Détecte le GPU sur macOS (Apple Silicon ou AMD)."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            displays = data.get("SPDisplaysDataType", [])
            if displays:
                gpu = displays[0]
                gpu_info = {
                    "type": "Apple" if platform.machine() == "arm64" else "Discrete/Integrated",
                    "name": gpu.get("sppci_model", "Unknown"),
                    "backend": "metal",
                }
                # VRAM ou mémoire unifiée
                vram = gpu.get("spdisplays_vram", gpu.get("spdisplays_vram_shared", ""))
                if vram:
                    gpu_info["vram"] = vram
                # Metal support
                metal_support = gpu.get("sppci_metal", gpu.get("spdisplays_metal", ""))
                gpu_info["metal_support"] = metal_support

                # Si Apple Silicon, indiquer la mémoire unifiée
                if platform.machine() == "arm64":
                    gpu_info["unified_memory"] = True
                    import psutil
                    total_ram = psutil.virtual_memory().total / (1024**3)
                    gpu_info["unified_memory_gb"] = round(total_ram, 1)

                return gpu_info
    except Exception:
        pass
    return None


def _detect_python_backends() -> Dict[str, bool]:
    """Vérifie quels backends Python ML sont disponibles."""
    backends = {}

    # PyTorch + CUDA
    try:
        import torch
        backends["pytorch"] = True
        backends["pytorch_version"] = torch.__version__
        backends["pytorch_cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            backends["pytorch_cuda_version"] = torch.version.cuda
        # MPS (Metal Performance Shaders)
        backends["pytorch_mps"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        backends["pytorch"] = False

    # llama-cpp-python
    try:
        import llama_cpp
        backends["llama_cpp"] = True
        backends["llama_cpp_version"] = getattr(llama_cpp, "__version__", "unknown")
    except ImportError:
        backends["llama_cpp"] = False

    return backends


def detect_ram() -> Dict[str, Any]:
    """Détecte les informations mémoire RAM."""
    import psutil

    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()

    info = {
        "total_gb": round(vm.total / (1024**3), 2),
        "available_gb": round(vm.available / (1024**3), 2),
        "used_gb": round(vm.used / (1024**3), 2),
        "percent_used": vm.percent,
        "swap_total_gb": round(swap.total / (1024**3), 2),
        "swap_used_gb": round(swap.used / (1024**3), 2),
    }

    # Mémoire unifiée sur Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        info["unified_memory"] = True
        info["note"] = "Apple Silicon utilise une mémoire unifiée partagée entre CPU et GPU."
    else:
        info["unified_memory"] = False

    # Type de mémoire (si disponible)
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                info["hw_memsize_bytes"] = int(result.stdout.strip())
        except Exception:
            pass

    return info


def get_full_hardware_info() -> Dict[str, Any]:
    """
    Retourne un rapport complet de la détection matérielle.
    Point d'entrée principal du module.
    """
    return {
        "os": detect_os(),
        "cpu": detect_cpu(),
        "gpu": detect_gpu(),
        "ram": detect_ram(),
    }


def get_hardware_summary(hw_info: Dict[str, Any]) -> str:
    """Génère un résumé lisible des informations matérielles."""
    os_info = hw_info["os"]
    cpu_info = hw_info["cpu"]
    gpu_info = hw_info["gpu"]
    ram_info = hw_info["ram"]

    lines = [
        "=" * 60,
        "  RAPPORT MATÉRIEL - ComputeLLM",
        "=" * 60,
        "",
        f"  OS         : {os_info['system']} {os_info['release']} ({os_info['architecture']})",
        f"  Python     : {os_info['python_version']}",
        "",
        f"  CPU        : {cpu_info.get('model', 'Unknown')}",
        f"  Cœurs      : {cpu_info.get('physical_cores', '?')} physiques / {cpu_info.get('logical_cores', '?')} logiques",
        f"  Arch       : {cpu_info.get('architecture_type', cpu_info.get('architecture', '?'))}",
    ]

    if cpu_info.get("is_apple_silicon"):
        perf = cpu_info.get("performance_cores", "?")
        eff = cpu_info.get("efficiency_cores", "?")
        lines.append(f"  Cœurs P/E  : {perf} performance / {eff} efficience")

    if cpu_info.get("frequency_mhz"):
        freq = cpu_info["frequency_mhz"]
        lines.append(f"  Fréquence  : {freq.get('current', '?')} MHz")

    lines.append("")
    lines.append(f"  RAM        : {ram_info['total_gb']} Go total / {ram_info['available_gb']} Go disponible")
    if ram_info.get("unified_memory"):
        lines.append(f"  Mémoire    : Unifiée (partagée CPU/GPU)")

    lines.append("")
    if gpu_info["gpus"]:
        for gpu in gpu_info["gpus"]:
            lines.append(f"  GPU        : {gpu['name']} ({gpu['type']})")
            if "vram_total_mb" in gpu:
                lines.append(f"  VRAM       : {gpu['vram_total_mb']:.0f} Mo")
            elif "unified_memory_gb" in gpu:
                lines.append(f"  Mém. GPU   : {gpu['unified_memory_gb']} Go (unifiée)")
    else:
        lines.append("  GPU        : Aucun détecté")

    lines.append(f"  Backend    : {gpu_info['primary_backend'].upper()}")
    lines.append(f"  Backends   : {', '.join(gpu_info['backends'])}")
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


if __name__ == "__main__":
    info = get_full_hardware_info()
    print(get_hardware_summary(info))
    print("\n[Détails JSON]")
    print(json.dumps(info, indent=2, default=str))
