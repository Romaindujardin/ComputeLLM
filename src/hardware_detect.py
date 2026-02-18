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
    Détecte tous les GPUs et les backends disponibles.
    Supporte NVIDIA (CUDA), AMD (ROCm), Intel (SYCL), Apple (Metal), et CPU fallback.
    """
    info: Dict[str, Any] = {
        "gpus": [],
        "backends": [],
        "primary_backend": "cpu",
    }

    system = platform.system()

    # --- Détection NVIDIA / CUDA ---
    nvidia_gpus = _detect_nvidia_gpu()
    if nvidia_gpus:
        info["gpus"].extend(nvidia_gpus)
        if "cuda" not in info["backends"]:
            info["backends"].append("cuda")
        info["primary_backend"] = "cuda"

    # --- Détection AMD / ROCm ou DirectML ---
    amd_gpus = _detect_amd_gpu()
    if amd_gpus:
        info["gpus"].extend(amd_gpus)
        # Déterminer le backend selon ce que les GPUs reportent
        amd_backend = amd_gpus[0].get("backend", "rocm")
        if amd_backend not in info["backends"]:
            info["backends"].append(amd_backend)
        if info["primary_backend"] == "cpu":
            info["primary_backend"] = amd_backend

    # --- Détection Intel GPU / SYCL ---
    intel_gpus = _detect_intel_gpu()
    if intel_gpus:
        info["gpus"].extend(intel_gpus)
        if "sycl" not in info["backends"]:
            info["backends"].append("sycl")
        if info["primary_backend"] == "cpu":
            info["primary_backend"] = "sycl"

    # --- Détection Metal (macOS) ---
    if system == "Darwin":
        metal_gpu = _detect_metal_gpu()
        if metal_gpu:
            info["gpus"].append(metal_gpu)
            info["backends"].append("metal")
            if info["primary_backend"] == "cpu":
                info["primary_backend"] = "metal"

    # --- Ajouter un index global à chaque GPU ---
    for idx, gpu in enumerate(info["gpus"]):
        gpu["gpu_index"] = idx

    # --- CPU fallback toujours disponible ---
    info["backends"].append("cpu")

    # --- Vérification des bibliothèques Python ---
    info["python_backends"] = _detect_python_backends()

    return info


def detect_all_gpus() -> list:
    """
    Retourne une liste plate de tous les GPUs détectés, chacun avec un champ
    'gpu_index' (0-based). Utile pour les sélecteurs UI.
    """
    gpu_info = detect_gpu()
    return gpu_info.get("gpus", [])


def _detect_nvidia_gpu() -> list:
    """
    Détecte tous les GPUs NVIDIA via nvidia-smi.
    Retourne une liste de dicts (un par GPU).
    """
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
                    gpus.append({
                        "type": "NVIDIA",
                        "name": parts[0],
                        "vram_total_mb": float(parts[1]),
                        "vram_free_mb": float(parts[2]),
                        "driver_version": parts[3],
                        "compute_capability": parts[4],
                        "backend": "cuda",
                        "device_index": len(gpus),
                    })
            if gpus:
                return gpus
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return []


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


def _detect_amd_gpu() -> list:
    """
    Détecte tous les GPUs AMD (Radeon RX, Radeon Pro, Instinct) via
    plusieurs méthodes : PyTorch ROCm, rocm-smi, lspci, WMI (Windows).
    Retourne une liste de dicts.
    """
    system = platform.system()

    # Méthode 1 : PyTorch ROCm (torch.version.hip) — énumère tous les devices
    try:
        import torch
        if torch.cuda.is_available() and hasattr(torch.version, "hip") and torch.version.hip:
            gpus = []
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                name = "AMD GPU"
                try:
                    name = torch.cuda.get_device_name(i)
                except Exception:
                    pass
                gpu_info = {
                    "type": "AMD",
                    "name": name,
                    "backend": "rocm",
                    "detected_via": "pytorch_rocm",
                    "hip_version": torch.version.hip,
                    "device_index": i,
                }
                try:
                    mem_total = torch.cuda.get_device_properties(i).total_mem
                    gpu_info["vram_total_mb"] = round(mem_total / (1024**2))
                except Exception:
                    pass
                gpus.append(gpu_info)
            if gpus:
                return gpus
    except ImportError:
        pass

    # Méthode 2 : rocm-smi (Linux avec pilotes AMD ROCm)
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram",
             "--showdriverversion", "--csv"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            output = result.stdout
            gpu_info = {
                "type": "AMD",
                "name": "AMD GPU",
                "backend": "rocm",
                "detected_via": "rocm-smi",
            }
            # Parser le CSV rocm-smi
            for line in output.strip().split("\n"):
                lower = line.lower()
                if "card series" in lower or "product name" in lower:
                    parts = line.split(",")
                    if len(parts) >= 2 and parts[1].strip():
                        gpu_info["name"] = parts[1].strip()
                if "vram total" in lower:
                    parts = line.split(",")
                    if len(parts) >= 2:
                        try:
                            vram_bytes = int(parts[1].strip())
                            gpu_info["vram_total_mb"] = round(vram_bytes / (1024**2))
                        except (ValueError, IndexError):
                            pass
                if "driver version" in lower:
                    parts = line.split(",")
                    if len(parts) >= 2 and parts[1].strip():
                        gpu_info["driver_version"] = parts[1].strip()
            return [gpu_info]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Méthode 2b : rocm-smi format texte simple (fallback)
    try:
        result = subprocess.run(
            ["rocm-smi"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and "GPU" in result.stdout:
            gpu_info = {
                "type": "AMD",
                "name": "AMD GPU (ROCm)",
                "backend": "rocm",
                "detected_via": "rocm-smi",
            }
            name_result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True, text=True, timeout=10,
            )
            if name_result.returncode == 0:
                for lr in name_result.stdout.strip().split("\n"):
                    if ":" in lr and ("card" in lr.lower() or "series" in lr.lower()):
                        gpu_info["name"] = lr.split(":", 1)[1].strip()
                        break
            return [gpu_info]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Méthode 3 : lspci (Linux) — collecte tous les GPUs AMD
    amd_gpu_keywords = ["radeon", "navi", "vega", "polaris", "ellesmere",
                        "instinct", "rx ", "w6", "w7", "firepro"]
    if system == "Linux":
        try:
            result = subprocess.run(
                ["lspci", "-nn"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split("\n"):
                    lower = line.lower()
                    if ("vga" in lower or "3d" in lower or "display" in lower):
                        if ("amd" in lower or "ati" in lower or "advanced micro" in lower) \
                           and any(kw in lower for kw in amd_gpu_keywords):
                            name_match = re.search(r"(?:AMD|ATI|Advanced Micro Devices).*?(?:\[|$)", line, re.IGNORECASE)
                            name = name_match.group(0).rstrip("[").strip() if name_match else "AMD GPU"
                            gpus.append({
                                "type": "AMD",
                                "name": name,
                                "backend": "rocm",
                                "detected_via": "lspci",
                            })
                if gpus:
                    return gpus
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # Méthode 4 : WMI / PowerShell (Windows) — collecte tous les GPUs AMD
    if system == "Windows":
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion | ConvertTo-Json"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                if isinstance(data, dict):
                    data = [data]
                gpus = []
                for gpu in data:
                    name = gpu.get("Name", "").lower()
                    if ("amd" in name or "radeon" in name or "ati" in name) \
                       and any(kw in name for kw in amd_gpu_keywords):
                        gpu_info = {
                            "type": "AMD",
                            "name": gpu.get("Name", "AMD GPU"),
                            "backend": "directml",
                            "detected_via": "wmi",
                        }
                        adapter_ram = gpu.get("AdapterRAM")
                        if adapter_ram and adapter_ram > 0:
                            gpu_info["vram_total_mb"] = round(adapter_ram / (1024**2))
                        driver = gpu.get("DriverVersion")
                        if driver:
                            gpu_info["driver_version"] = driver
                        gpus.append(gpu_info)
                if gpus:
                    return gpus
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
            pass

    return []


def _detect_intel_gpu() -> list:
    """
    Détecte tous les GPUs Intel (Arc, Data Center GPU, Flex, intégré) via
    plusieurs méthodes : PyTorch XPU, xpu-smi, lspci, WMI (Windows).
    Retourne une liste de dicts.
    """
    system = platform.system()

    # Méthode 1 : PyTorch XPU — énumère tous les devices
    try:
        import torch
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            gpus = []
            device_count = torch.xpu.device_count()
            for i in range(device_count):
                name = "Intel GPU"
                try:
                    name = torch.xpu.get_device_name(i)
                except Exception:
                    pass
                gpu_info = {
                    "type": "Intel",
                    "name": name,
                    "backend": "sycl",
                    "detected_via": "pytorch_xpu",
                    "device_index": i,
                }
                try:
                    mem_total = torch.xpu.get_device_properties(i).total_memory
                    gpu_info["vram_total_mb"] = round(mem_total / (1024**2))
                except Exception:
                    pass
                gpus.append(gpu_info)
            if gpus:
                return gpus
    except ImportError:
        pass

    # Méthode 2 : xpu-smi (Linux avec pilotes Intel)
    if system == "Linux":
        try:
            result = subprocess.run(
                ["xpu-smi", "discovery"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                output = result.stdout
                name_match = re.search(r"Device Name\s*:\s*(.+)", output)
                mem_match = re.search(r"Memory Physical Size\s*:\s*([\d.]+)\s*(\w+)", output)
                name = name_match.group(1).strip() if name_match else "Intel GPU"
                gpu_info = {
                    "type": "Intel",
                    "name": name,
                    "backend": "sycl",
                    "detected_via": "xpu-smi",
                }
                if mem_match:
                    size_val = float(mem_match.group(1))
                    unit = mem_match.group(2).upper()
                    if "GI" in unit or "GB" in unit:
                        gpu_info["vram_total_mb"] = round(size_val * 1024)
                    elif "MI" in unit or "MB" in unit:
                        gpu_info["vram_total_mb"] = round(size_val)
                return [gpu_info]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # Méthode 3 : lspci (Linux) ou WMI (Windows)
    intel_gpu_keywords = ["arc ", "dg1", "dg2", "flex ", "iris", "uhd graphics"]

    if system == "Linux":
        try:
            result = subprocess.run(
                ["lspci", "-nn"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split("\n"):
                    lower = line.lower()
                    if "vga" in lower or "3d" in lower or "display" in lower:
                        if "intel" in lower and any(kw in lower for kw in intel_gpu_keywords):
                            name_match = re.search(r"Intel.*?(?:\[|$)", line)
                            name = name_match.group(0).rstrip("[").strip() if name_match else "Intel GPU"
                            gpus.append({
                                "type": "Intel",
                                "name": name,
                                "backend": "sycl",
                                "detected_via": "lspci",
                            })
                if gpus:
                    return gpus
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    elif system == "Windows":
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM | ConvertTo-Json"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                if isinstance(data, dict):
                    data = [data]
                gpus = []
                for gpu in data:
                    name = gpu.get("Name", "").lower()
                    if "intel" in name and any(kw in name for kw in intel_gpu_keywords):
                        gpu_info = {
                            "type": "Intel",
                            "name": gpu.get("Name", "Intel GPU"),
                            "backend": "sycl",
                            "detected_via": "wmi",
                        }
                        adapter_ram = gpu.get("AdapterRAM")
                        if adapter_ram and adapter_ram > 0:
                            gpu_info["vram_total_mb"] = round(adapter_ram / (1024**2))
                        gpus.append(gpu_info)
                if gpus:
                    return gpus
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
            pass

    return []


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
        # XPU (Intel)
        backends["pytorch_xpu"] = hasattr(torch, "xpu") and torch.xpu.is_available()
        if backends["pytorch_xpu"]:
            try:
                backends["pytorch_xpu_device"] = torch.xpu.get_device_name(0)
            except Exception:
                pass
    except ImportError:
        backends["pytorch"] = False

    # llama-cpp-python
    try:
        import llama_cpp
        backends["llama_cpp"] = True
        backends["llama_cpp_version"] = getattr(llama_cpp, "__version__", "unknown")
    except ImportError:
        backends["llama_cpp"] = False

    # llama-server (binaire)
    try:
        from src.llama_server import find_llama_server_binary
        server_path = find_llama_server_binary()
        backends["llama_server"] = server_path is not None
        if server_path:
            backends["llama_server_path"] = server_path
    except ImportError:
        backends["llama_server"] = False

    # ROCm (AMD) — torch.version.hip est set si PyTorch est compilé avec ROCm
    try:
        import torch
        if hasattr(torch.version, "hip") and torch.version.hip:
            backends["pytorch_rocm"] = True
            backends["pytorch_hip_version"] = torch.version.hip
        else:
            backends["pytorch_rocm"] = False
    except (ImportError, Exception):
        if "pytorch_rocm" not in backends:
            backends["pytorch_rocm"] = False

    # Intel Extension for PyTorch (IPEX)
    try:
        import intel_extension_for_pytorch
        backends["ipex"] = True
        backends["ipex_version"] = getattr(intel_extension_for_pytorch, "__version__", "unknown")
    except ImportError:
        backends["ipex"] = False

    # DirectML (Windows — AMD/Intel/NVIDIA via DirectX 12)
    try:
        import torch_directml
        backends["directml"] = True
        backends["directml_available"] = torch_directml.is_available()
        backends["directml_device_count"] = torch_directml.device_count()
        if torch_directml.device_count() > 0:
            backends["directml_device_name"] = torch_directml.device_name(0)
    except ImportError:
        backends["directml"] = False

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
