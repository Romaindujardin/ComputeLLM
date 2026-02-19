"""
ComputeLLM - Module de benchmark IA (inférence LLM locale).
Gère le téléchargement des modèles, l'exécution de l'inférence
et la mesure des métriques de performance.
"""

import time
import os
import platform
import gc
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

import numpy as np
import psutil

from src.config import (
    AVAILABLE_MODELS,
    MODELS_DIR,
    BENCHMARK_PROMPT,
    INFERENCE_CONFIG,
    QUANTIZATION_VARIANTS,
    TEMPERATURE_VARIANTS,
    LANGUAGE_PROMPTS,
    PROMPT_TYPE_VARIANTS,
)
from src.benchmark_classic import ResourceMonitor, system_cleanup


# =============================================================================
# Gestion des modèles (téléchargement via HuggingFace Hub)
# =============================================================================

def list_available_models() -> Dict[str, Dict]:
    """Retourne la liste des modèles disponibles pour le benchmark."""
    return AVAILABLE_MODELS


def get_compatible_models(ram_gb: float) -> Dict[str, Dict]:
    """
    Retourne les modèles compatibles avec la RAM disponible.
    
    Args:
        ram_gb: RAM disponible en Go.
    """
    compatible = {}
    for key, model in AVAILABLE_MODELS.items():
        if model["min_ram_gb"] <= ram_gb:
            compatible[key] = model
    return compatible


def is_model_downloaded(model_key: str) -> bool:
    """Vérifie si un modèle est déjà téléchargé localement."""
    if model_key not in AVAILABLE_MODELS:
        return False
    model = AVAILABLE_MODELS[model_key]
    filepath = MODELS_DIR / model["filename"]
    return filepath.exists()


def get_model_path(model_key: str) -> Optional[Path]:
    """Retourne le chemin local du modèle s'il est téléchargé."""
    if model_key not in AVAILABLE_MODELS:
        return None
    model = AVAILABLE_MODELS[model_key]
    filepath = MODELS_DIR / model["filename"]
    if filepath.exists():
        return filepath
    return None


# =============================================================================
# Gestion des quantifications
# =============================================================================

def get_available_quantizations(model_key: str) -> Dict[str, Dict]:
    """
    Retourne les variantes de quantification disponibles pour un modèle.

    Args:
        model_key: Clé du modèle dans AVAILABLE_MODELS.

    Returns:
        Dict des variantes de quantification {quant_key: info_dict}.
    """
    return QUANTIZATION_VARIANTS.get(model_key, {})


def get_compatible_quantizations(model_key: str, ram_gb: float) -> Dict[str, Dict]:
    """
    Retourne les quantifications compatibles avec la RAM disponible.

    Args:
        model_key: Clé du modèle.
        ram_gb: RAM totale du système en Go.

    Returns:
        Dict des variantes compatibles.
    """
    variants = get_available_quantizations(model_key)
    return {
        qk: qv for qk, qv in variants.items()
        if qv["min_ram_gb"] <= ram_gb
    }


def is_quantization_downloaded(model_key: str, quant_key: str) -> bool:
    """Vérifie si une quantification spécifique est déjà téléchargée."""
    variants = get_available_quantizations(model_key)
    if quant_key not in variants:
        return False
    filepath = MODELS_DIR / variants[quant_key]["filename"]
    return filepath.exists()


def download_quantization(
    model_key: str,
    quant_key: str,
    progress_callback: Optional[Callable] = None,
) -> Path:
    """
    Télécharge une variante de quantification spécifique.

    Args:
        model_key: Clé du modèle dans AVAILABLE_MODELS.
        quant_key: Clé de la quantification (ex: "Q4_K_M").
        progress_callback: Callback(progress, message).

    Returns:
        Path vers le fichier modèle téléchargé.
    """
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Modèle inconnu : {model_key}")

    variants = get_available_quantizations(model_key)
    if quant_key not in variants:
        raise ValueError(f"Quantification inconnue : {quant_key} pour {model_key}")

    model = AVAILABLE_MODELS[model_key]
    variant = variants[quant_key]
    local_path = MODELS_DIR / variant["filename"]

    if local_path.exists():
        if progress_callback:
            progress_callback(1.0, f"{model['name']} {quant_key} déjà téléchargé.")
        return local_path

    if progress_callback:
        progress_callback(0.0, f"Téléchargement de {model['name']} {quant_key} ({variant['size_gb']} Go)...")

    try:
        from huggingface_hub import hf_hub_download

        downloaded_path = hf_hub_download(
            repo_id=model["repo_id"],
            filename=variant["filename"],
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False,
        )

        if progress_callback:
            progress_callback(1.0, f"{model['name']} {quant_key} téléchargé.")

        return Path(downloaded_path)

    except ImportError:
        raise ImportError(
            "huggingface_hub n'est pas installé. "
            "Installez-le avec : pip install huggingface_hub"
        )
    except Exception as e:
        raise RuntimeError(
            f"Erreur téléchargement {model['name']} {quant_key}: {e}"
        )


def delete_quantization(model_key: str, quant_key: str) -> bool:
    """
    Supprime une variante de quantification téléchargée.

    Args:
        model_key: Clé du modèle.
        quant_key: Clé de la quantification (ex: "Q4_K_M").

    Returns:
        True si le fichier a été supprimé, False sinon.
    """
    variants = get_available_quantizations(model_key)
    if quant_key not in variants:
        return False
    filepath = MODELS_DIR / variants[quant_key]["filename"]
    if filepath.exists():
        filepath.unlink()
        return True
    return False


def download_model(
    model_key: str,
    progress_callback: Optional[Callable] = None,
) -> Path:
    """
    Télécharge un modèle depuis HuggingFace Hub.
    
    Args:
        model_key: Clé du modèle dans AVAILABLE_MODELS.
        progress_callback: Callback(progress, message) pour le suivi.
    
    Returns:
        Path vers le fichier modèle téléchargé.
    """
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Modèle inconnu : {model_key}. Modèles disponibles : {list(AVAILABLE_MODELS.keys())}")

    model = AVAILABLE_MODELS[model_key]
    local_path = MODELS_DIR / model["filename"]

    # Vérifier si déjà téléchargé
    if local_path.exists():
        if progress_callback:
            progress_callback(1.0, f"Modèle {model['name']} déjà téléchargé.")
        return local_path

    if progress_callback:
        progress_callback(0.0, f"Téléchargement de {model['name']} ({model['size_gb']} Go)...")

    try:
        from huggingface_hub import hf_hub_download

        # Télécharger le fichier GGUF
        downloaded_path = hf_hub_download(
            repo_id=model["repo_id"],
            filename=model["filename"],
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False,
        )

        if progress_callback:
            progress_callback(1.0, f"Modèle {model['name']} téléchargé avec succès.")

        return Path(downloaded_path)

    except ImportError:
        raise ImportError(
            "huggingface_hub n'est pas installé. "
            "Installez-le avec : pip install huggingface_hub"
        )
    except Exception as e:
        raise RuntimeError(f"Erreur lors du téléchargement de {model['name']}: {e}")


# =============================================================================
# Détection du backend optimal
# =============================================================================

def detect_best_backend(selected_gpu: Dict[str, Any] = None, inference_mode: str = "local") -> Dict[str, Any]:
    """
    Détecte le meilleur backend disponible pour l'inférence.
    Retourne des informations sur le backend sélectionné.
    Supporte : CUDA (NVIDIA), ROCm (AMD), SYCL/XPU (Intel), Metal (Apple), CPU fallback.

    Args:
        selected_gpu: (Optionnel) Dict GPU provenant de detect_all_gpus().
                     Si fourni, utilise ce GPU au lieu de l'auto-détection.
        inference_mode: "local" (llama-cpp-python) ou "server" (llama-server).
                       En mode serveur, on peut utiliser des backends non supportés
                       par les bindings Python (ex: Vulkan pour AMD sur Windows).
    """
    system = platform.system()
    machine = platform.machine()

    result = {
        "backend": "cpu",
        "n_gpu_layers": 0,
        "details": "",
    }

    # Si un GPU est explicitement sélectionné par l'utilisateur
    if selected_gpu is not None:
        backend = selected_gpu.get("backend", "cpu")
        gpu_name = selected_gpu.get("name", "GPU")

        if backend == "cuda":
            result["backend"] = "cuda"
            result["n_gpu_layers"] = -1
            result["details"] = f"GPU sélectionné : {gpu_name} (CUDA)"
            return result
        elif backend == "rocm":
            result["backend"] = "rocm"
            result["n_gpu_layers"] = -1
            result["details"] = f"GPU sélectionné : {gpu_name} (ROCm/HIP)"
            result["amd_device"] = gpu_name
            if "vram_total_mb" in selected_gpu:
                result["amd_vram_mb"] = selected_gpu["vram_total_mb"]
            if "hip_version" in selected_gpu:
                result["hip_version"] = selected_gpu["hip_version"]
            if "driver_version" in selected_gpu:
                result["driver_version"] = selected_gpu["driver_version"]
            return result
        elif backend == "sycl":
            result["backend"] = "sycl"
            result["n_gpu_layers"] = -1
            result["details"] = f"GPU sélectionné : {gpu_name} (SYCL)"
            result["intel_device"] = gpu_name
            if "vram_total_mb" in selected_gpu:
                result["intel_vram_mb"] = selected_gpu["vram_total_mb"]
            return result
        elif backend == "metal":
            result["backend"] = "metal"
            result["n_gpu_layers"] = -1
            result["details"] = f"GPU sélectionné : {gpu_name} (Metal)"
            return result
        elif backend == "directml":
            # DirectML (Windows AMD)
            if inference_mode == "server":
                # Mode Serveur : On suppose que le binaire supporte Vulkan (ou équivalent)
                result["backend"] = "vulkan"  # Ou "gpu" générique
                result["n_gpu_layers"] = -1
                result["details"] = f"GPU sélectionné : {gpu_name} (Mode Serveur/Vulkan)"
                return result
            else:
                # Mode Local (Python) : llama-cpp-python ne supporte pas DirectML
                result["backend"] = "cpu"
                result["n_gpu_layers"] = 0
                result["details"] = (
                    f"GPU sélectionné : {gpu_name} (DirectML) — "
                    f"llama-cpp-python ne supporte pas DirectML. "
                    f"Utilisez le mode llama-server avec un binaire Vulkan pour "
                    f"l'accélération GPU."
                )
                return result
        else:
            # CPU explicitement choisi
            result["details"] = f"CPU sélectionné (pas d'accélération GPU)"
            return result

    # === Auto-détection (comportement original) ===

    # Vérifier CUDA / ROCm (NVIDIA ou AMD)
    try:
        import subprocess
        nvidia_check = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5
        )
        if nvidia_check.returncode == 0:
            result["backend"] = "cuda"
            result["n_gpu_layers"] = -1  # Toutes les couches sur GPU
            result["details"] = "NVIDIA GPU détecté, utilisation de CUDA"
            return result
    except (FileNotFoundError, Exception):
        pass

    # Vérifier AMD GPU / ROCm (HIP) ou DirectML (Windows)
    # Réutilise _detect_amd_gpu() de hardware_detect pour la cohérence
    try:
        from src.hardware_detect import _detect_amd_gpu
        amd_gpus = _detect_amd_gpu()
        if amd_gpus:
            amd_gpu = amd_gpus[0]  # Prendre le premier par défaut
            gpu_name = amd_gpu.get("name", "AMD GPU")
            detected_via = amd_gpu.get("detected_via", "unknown")
            gpu_backend = amd_gpu.get("backend", "rocm")

            if gpu_backend == "directml":
                # Windows AMD — DirectML n'est pas supporté par llama-cpp-python
                result["backend"] = "cpu"
                result["n_gpu_layers"] = 0
                result["details"] = (
                    f"AMD GPU détecté ({gpu_name}) — DirectML n'est pas supporté "
                    f"par llama-cpp-python. Utilisez le mode llama-server avec un "
                    f"binaire Vulkan pour l'accélération GPU, sinon fallback CPU."
                )
                return result
            else:
                # Linux/WSL — ROCm disponible
                result["backend"] = "rocm"
                result["n_gpu_layers"] = -1
                result["details"] = f"AMD GPU détecté ({gpu_name} via {detected_via}), utilisation de ROCm/HIP"
                result["amd_device"] = gpu_name
                if "vram_total_mb" in amd_gpu:
                    result["amd_vram_mb"] = amd_gpu["vram_total_mb"]
                if "hip_version" in amd_gpu:
                    result["hip_version"] = amd_gpu["hip_version"]
                if "driver_version" in amd_gpu:
                    result["driver_version"] = amd_gpu["driver_version"]
                return result
    except Exception:
        pass

    # Vérifier Intel GPU / SYCL (Windows/Linux)
    # Réutilise _detect_intel_gpu() de hardware_detect pour la cohérence
    try:
        from src.hardware_detect import _detect_intel_gpu
        intel_gpus = _detect_intel_gpu()
        if intel_gpus:
            intel_gpu = intel_gpus[0]  # Prendre le premier par défaut
            gpu_name = intel_gpu.get("name", "Intel GPU")
            detected_via = intel_gpu.get("detected_via", "unknown")
            result["backend"] = "sycl"
            result["n_gpu_layers"] = -1
            result["details"] = f"Intel GPU détecté ({gpu_name} via {detected_via}), utilisation de SYCL"
            result["intel_device"] = gpu_name
            if "vram_total_mb" in intel_gpu:
                result["intel_vram_mb"] = intel_gpu["vram_total_mb"]
            return result
    except Exception:
        pass

    # Vérifier Metal (macOS)
    if system == "Darwin":
        if machine == "arm64":
            result["backend"] = "metal"
            result["n_gpu_layers"] = -1  # Toutes les couches sur GPU
            result["details"] = "Apple Silicon détecté, utilisation de Metal"
        else:
            # Intel Mac - vérifier si Metal est supporté
            result["backend"] = "metal"
            result["n_gpu_layers"] = -1
            result["details"] = "macOS détecté, tentative Metal"
        return result

    result["details"] = "Aucun accélérateur détecté, fallback CPU"
    return result


# =============================================================================
# Inférence LLM avec llama-cpp-python
# =============================================================================

def _load_model(model_path: str, backend_info: Dict[str, Any], n_ctx: int = 2048):
    """
    Charge un modèle GGUF avec llama-cpp-python.
    
    Returns:
        Instance Llama.
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python n'est pas installé.\n"
            "Installation :\n"
            "  macOS (Metal) : CMAKE_ARGS=\"-DGGML_METAL=on\" pip install llama-cpp-python\n"
            "  Windows (CUDA): CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python\n"
            "  AMD (ROCm)    : CMAKE_ARGS=\"-DGGML_HIPBLAS=on\" pip install llama-cpp-python\n"
            "  Intel (SYCL)  : CMAKE_ARGS=\"-DGGML_SYCL=on\" pip install llama-cpp-python\n"
            "  CPU only      : pip install llama-cpp-python\n"
            "\n"
            "Alternative sans compilation : utilisez le mode llama-server\n"
            "  Téléchargez le binaire depuis https://github.com/ggerganov/llama.cpp/releases\n"
            "  Puis activez 'Utiliser llama-server' dans le Benchmark."
        )

    n_gpu_layers = backend_info.get("n_gpu_layers", 0)

    model = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        seed=INFERENCE_CONFIG["seed"],
        verbose=False,
    )

    return model


def run_single_inference(
    model,
    prompt: str = BENCHMARK_PROMPT,
    max_tokens: int = None,
    temperature: float = None,
) -> Dict[str, Any]:
    """
    Exécute une inférence unique et mesure les métriques.
    Utilise le format chat (create_chat_completion) pour une meilleure
    compatibilité avec toutes les quantifications des modèles instruct/chat.
    
    Args:
        model: Instance Llama chargée.
        prompt: Texte du prompt utilisateur.
        max_tokens: Nombre max de tokens à générer.
        temperature: Température de génération (override INFERENCE_CONFIG si fourni).
    
    Returns:
        Dict avec latence premier token, tokens/s, mémoire, etc.
    """
    if max_tokens is None:
        max_tokens = INFERENCE_CONFIG["max_tokens"]
    if temperature is None:
        temperature = INFERENCE_CONFIG["temperature"]

    # Mesurer la mémoire avant
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024**3)  # En Go

    # Mesurer l'inférence avec streaming pour capturer le premier token
    tokens_generated = 0
    first_token_time = None
    token_times = []
    error = None

    try:
        start_time = time.perf_counter()

        # Utiliser create_chat_completion (format chat) pour compatibilité
        # avec toutes les quantifications des modèles instruct/chat.
        stream = model.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=INFERENCE_CONFIG["top_p"],
            repeat_penalty=INFERENCE_CONFIG["repeat_penalty"],
            stream=True,
        )

        generated_text = ""
        for chunk in stream:
            current_time = time.perf_counter()
            delta = chunk["choices"][0].get("delta", {})
            token_text = delta.get("content", "")

            if token_text:  # Ignorer les tokens vides
                tokens_generated += 1
                token_times.append(current_time)

                if first_token_time is None:
                    first_token_time = current_time - start_time

                generated_text += token_text

        total_time = time.perf_counter() - start_time

    except Exception as e:
        error = str(e)
        total_time = time.perf_counter() - start_time

    # Marquer comme échoué si aucun token n'a été généré
    if tokens_generated == 0 and error is None:
        error = "Aucun token généré (le modèle a retourné un EOS immédiat)"

    # Mesurer la mémoire après
    mem_after = process.memory_info().rss / (1024**3)

    # Calculer les métriques
    result = {
        "tokens_generated": tokens_generated,
        "total_time_s": round(total_time, 4),
        "first_token_latency_s": round(first_token_time, 4) if first_token_time else None,
        "tokens_per_second": round(tokens_generated / total_time, 2) if total_time > 0 and tokens_generated > 0 else 0,
        "memory_before_gb": round(mem_before, 3),
        "memory_after_gb": round(mem_after, 3),
        "memory_delta_gb": round(mem_after - mem_before, 3),
        "error": error,
        "success": error is None,
    }

    # Calculer la latence inter-tokens si possible
    if len(token_times) > 1:
        inter_token_latencies = [
            token_times[i] - token_times[i-1]
            for i in range(1, len(token_times))
        ]
        result["avg_inter_token_latency_ms"] = round(
            np.mean(inter_token_latencies) * 1000, 2
        )
        result["p90_inter_token_latency_ms"] = round(
            np.percentile(inter_token_latencies, 90) * 1000, 2
        )

    return result


def benchmark_model(
    model_key: str,
    progress_callback: Optional[Callable] = None,
    selected_gpu: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Exécute le benchmark complet pour un modèle donné.
    Inclut le téléchargement, le chargement, l'échauffement et les runs de benchmark.
    """
    if model_key not in AVAILABLE_MODELS:
        return {"error": f"Modèle inconnu : {model_key}"}

    model_config = AVAILABLE_MODELS[model_key]

    # Nettoyage mémoire avant le benchmark
    if progress_callback:
        progress_callback(0.01, f"Nettoyage mémoire avant {model_config['name']}...")
    cleanup_info = system_cleanup(f"LLM {model_config['name']}")

    # Vérifier la RAM : on utilise la RAM totale du système (pas l'instantanée)
    # car macOS gère dynamiquement les caches et libère la mémoire à la demande.
    ram_total = psutil.virtual_memory().total / (1024**3)
    if ram_total < model_config["min_ram_gb"]:
        return {
            "model": model_config["name"],
            "status": "skipped",
            "reason": f"RAM totale insuffisante ({ram_total:.1f} Go total, "
                      f"{model_config['min_ram_gb']} Go requis)",
        }

    result = {
        "model": model_config["name"],
        "params": model_config["params"],
        "status": "running",
        "runs": [],
    }

    try:
        # Étape 1: Téléchargement
        if progress_callback:
            progress_callback(0.05, f"Vérification du modèle {model_config['name']}...")

        model_path = download_model(model_key, progress_callback=None)

        # Étape 2: Détection du backend
        backend_info = detect_best_backend(selected_gpu=selected_gpu)
        result["backend"] = backend_info

        if progress_callback:
            progress_callback(0.10, f"Chargement de {model_config['name']} ({backend_info['backend']})...")

        # Étape 3: Chargement du modèle
        load_start = time.perf_counter()
        model = _load_model(
            model_path,
            backend_info,
            n_ctx=INFERENCE_CONFIG["n_ctx"],
        )
        load_time = time.perf_counter() - load_start
        result["model_load_time_s"] = round(load_time, 2)

        # Étape 4: Échauffement
        if progress_callback:
            progress_callback(0.20, f"Échauffement {model_config['name']}...")

        for _ in range(INFERENCE_CONFIG["n_warmup_runs"]):
            run_single_inference(model, max_tokens=32)

        # Étape 5: Exécution des benchmarks
        n_runs = INFERENCE_CONFIG["n_benchmark_runs"]
        monitor = ResourceMonitor()
        monitor.start()

        for i in range(n_runs):
            if progress_callback:
                p = 0.30 + (i / n_runs) * 0.65
                progress_callback(p, f"Inférence {model_config['name']} - Run {i+1}/{n_runs}")

            run_result = run_single_inference(model)
            result["runs"].append(run_result)

        resource_usage = monitor.stop()
        result["resource_usage"] = resource_usage

        # Étape 6: Calcul des statistiques agrégées
        successful_runs = [r for r in result["runs"] if r["success"]]
        if successful_runs:
            result["summary"] = {
                "n_successful_runs": len(successful_runs),
                "n_total_runs": n_runs,
                "avg_tokens_per_second": round(
                    np.mean([r["tokens_per_second"] for r in successful_runs]), 2
                ),
                "std_tokens_per_second": round(
                    np.std([r["tokens_per_second"] for r in successful_runs]), 2
                ),
                "avg_first_token_latency_s": round(
                    np.mean([r["first_token_latency_s"] for r in successful_runs
                            if r["first_token_latency_s"] is not None]), 4
                ),
                "avg_total_time_s": round(
                    np.mean([r["total_time_s"] for r in successful_runs]), 4
                ),
                "peak_memory_gb": round(
                    max(r["memory_after_gb"] for r in successful_runs), 3
                ),
                "stability": "stable" if len(successful_runs) == n_runs else "unstable",
            }

        result["status"] = "completed"

        # Nettoyage
        del model
        gc.collect()

        if progress_callback:
            progress_callback(1.0, f"Benchmark {model_config['name']} terminé.")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        if progress_callback:
            progress_callback(1.0, f"Erreur : {e}")

    return result


# =============================================================================
# Benchmark comparatif de quantification
# =============================================================================

def benchmark_single_quantization(
    model_key: str,
    quant_key: str,
    progress_callback: Optional[Callable] = None,
    selected_gpu: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Exécute le benchmark pour une variante de quantification spécifique.

    Args:
        model_key: Clé du modèle de base.
        quant_key: Clé de quantification (ex: "Q4_K_M").
        progress_callback: Callback(progress, message).

    Returns:
        Dict avec résultats détaillés pour cette quantification.
    """
    if model_key not in AVAILABLE_MODELS:
        return {"error": f"Modèle inconnu : {model_key}"}

    variants = get_available_quantizations(model_key)
    if quant_key not in variants:
        return {"error": f"Quantification inconnue : {quant_key}"}

    model_config = AVAILABLE_MODELS[model_key]
    variant = variants[quant_key]

    # Nettoyage mémoire
    if progress_callback:
        progress_callback(0.01, f"Nettoyage avant {model_config['name']} {quant_key}...")
    cleanup_info = system_cleanup(f"Quant {model_config['name']} {quant_key}")

    # Vérifier la RAM
    ram_total = psutil.virtual_memory().total / (1024**3)
    if ram_total < variant["min_ram_gb"]:
        return {
            "quantization": quant_key,
            "bits": variant["bits"],
            "file_size_gb": variant["size_gb"],
            "status": "skipped",
            "reason": f"RAM insuffisante ({ram_total:.1f} Go, {variant['min_ram_gb']} Go requis)",
        }

    result = {
        "quantization": quant_key,
        "bits": variant["bits"],
        "file_size_gb": variant["size_gb"],
        "status": "running",
        "runs": [],
    }

    try:
        # Téléchargement / vérification
        if progress_callback:
            progress_callback(0.05, f"Vérification {model_config['name']} {quant_key}...")
        model_path = download_quantization(model_key, quant_key)

        # Mesurer la taille réelle du fichier
        actual_size_gb = model_path.stat().st_size / (1024**3)
        result["actual_file_size_gb"] = round(actual_size_gb, 3)

        # Backend
        backend_info = detect_best_backend(selected_gpu=selected_gpu)
        result["backend"] = backend_info

        if progress_callback:
            progress_callback(0.10, f"Chargement {model_config['name']} {quant_key} ({backend_info['backend']})...")

        # Chargement du modèle avec mesure du temps
        load_start = time.perf_counter()
        model = _load_model(
            model_path,
            backend_info,
            n_ctx=INFERENCE_CONFIG["n_ctx"],
        )
        load_time = time.perf_counter() - load_start
        result["model_load_time_s"] = round(load_time, 2)

        # Mémoire après chargement
        process = psutil.Process()
        result["memory_after_load_gb"] = round(
            process.memory_info().rss / (1024**3), 3
        )

        # Échauffement
        if progress_callback:
            progress_callback(0.20, f"Échauffement {quant_key}...")
        for _ in range(INFERENCE_CONFIG["n_warmup_runs"]):
            run_single_inference(model, max_tokens=32)

        # Runs de benchmark
        n_runs = INFERENCE_CONFIG["n_benchmark_runs"]
        monitor = ResourceMonitor()
        monitor.start()

        for i in range(n_runs):
            if progress_callback:
                p = 0.30 + (i / n_runs) * 0.65
                progress_callback(p, f"{quant_key} — Run {i+1}/{n_runs}")

            run_result = run_single_inference(model)
            result["runs"].append(run_result)

        resource_usage = monitor.stop()
        result["resource_usage"] = resource_usage

        # Statistiques agrégées
        successful_runs = [r for r in result["runs"] if r["success"]]
        if successful_runs:
            tps_values = [r["tokens_per_second"] for r in successful_runs]
            ftl_values = [
                r["first_token_latency_s"] for r in successful_runs
                if r["first_token_latency_s"] is not None
            ]
            itl_values = [
                r["avg_inter_token_latency_ms"] for r in successful_runs
                if "avg_inter_token_latency_ms" in r
            ]

            result["summary"] = {
                "n_successful_runs": len(successful_runs),
                "n_total_runs": n_runs,
                "avg_tokens_per_second": round(np.mean(tps_values), 2),
                "std_tokens_per_second": round(np.std(tps_values), 2),
                "min_tokens_per_second": round(min(tps_values), 2),
                "max_tokens_per_second": round(max(tps_values), 2),
                "avg_first_token_latency_s": round(np.mean(ftl_values), 4) if ftl_values else None,
                "avg_inter_token_latency_ms": round(np.mean(itl_values), 2) if itl_values else None,
                "avg_total_time_s": round(
                    np.mean([r["total_time_s"] for r in successful_runs]), 4
                ),
                "peak_memory_gb": round(
                    max(r["memory_after_gb"] for r in successful_runs), 3
                ),
                "stability": "stable" if len(successful_runs) == n_runs else "unstable",
            }

        result["status"] = "completed"

        # Nettoyage modèle
        del model
        gc.collect()

        if progress_callback:
            progress_callback(1.0, f"{quant_key} terminé.")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        if progress_callback:
            progress_callback(1.0, f"Erreur {quant_key}: {e}")

    return result


def run_quantization_comparison(
    model_key: str,
    quant_keys: List[str],
    progress_callback: Optional[Callable] = None,
    selected_gpu: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Exécute un benchmark comparatif de toutes les quantifications sélectionnées
    pour un même modèle.

    Args:
        model_key: Clé du modèle de base (ex: "tinyllama-1.1b").
        quant_keys: Liste des quantifications à tester (ex: ["Q2_K", "Q4_K_M", "Q8_0"]).
        progress_callback: Callback(progress, message).

    Returns:
        Dict structuré avec résultats par quantification et comparaison.
    """
    if model_key not in AVAILABLE_MODELS:
        return {"error": f"Modèle inconnu : {model_key}"}

    model_config = AVAILABLE_MODELS[model_key]
    start_time = time.time()
    n_quants = len(quant_keys)

    comparison = {
        "model_name": model_config["name"],
        "model_key": model_key,
        "params": model_config["params"],
        "variants_tested": quant_keys,
        "results": {},
    }

    for idx, quant_key in enumerate(quant_keys):
        def quant_progress(p, msg, _idx=idx):
            if progress_callback:
                overall = (_idx + p) / n_quants
                progress_callback(overall, msg)

        if progress_callback:
            progress_callback(
                idx / n_quants,
                f"Quantification {idx+1}/{n_quants} : {model_config['name']} {quant_key}",
            )

        # Nettoyage mémoire entre chaque quantification
        if idx > 0:
            system_cleanup(f"entre quantifications ({quant_key})")

        comparison["results"][quant_key] = benchmark_single_quantization(
            model_key,
            quant_key,
            progress_callback=quant_progress,
            selected_gpu=selected_gpu,
        )

    elapsed = time.time() - start_time
    comparison["total_time_s"] = round(elapsed, 2)

    # Construire un tableau comparatif synthétique
    summary_table = []
    for qk in quant_keys:
        qr = comparison["results"].get(qk, {})
        summary = qr.get("summary", {})
        if summary:
            summary_table.append({
                "quantization": qk,
                "bits": qr.get("bits", 0),
                "file_size_gb": qr.get("actual_file_size_gb", qr.get("file_size_gb", 0)),
                "tokens_per_second": summary.get("avg_tokens_per_second", 0),
                "first_token_latency_s": summary.get("avg_first_token_latency_s", 0),
                "inter_token_latency_ms": summary.get("avg_inter_token_latency_ms", 0),
                "peak_memory_gb": summary.get("peak_memory_gb", 0),
                "model_load_time_s": qr.get("model_load_time_s", 0),
                "stability": summary.get("stability", "unknown"),
            })
    comparison["comparison_table"] = summary_table

    return comparison


def run_all_ai_benchmarks(
    model_keys: List[str] = None,
    quantization_models: Dict[str, List[str]] = None,
    temperature_models: Dict[str, List[str]] = None,
    language_models: Dict[str, List[str]] = None,
    prompt_type_models: Dict[str, List[str]] = None,
    progress_callback: Optional[Callable] = None,
    selected_gpu: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Exécute les benchmarks IA pour tous les modèles sélectionnés,
    incluant optionnellement la comparaison de quantification,
    température, langue et type de prompt.

    Args:
        model_keys: Liste des clés de modèles à tester.
        quantization_models: Dict {model_key: [quant_key, ...]} pour
            le comparatif de quantification. Si None, pas de comparatif.
        temperature_models: Dict {model_key: [temp_key, ...]} pour
            le comparatif de température. Si None, pas de comparatif.
        language_models: Dict {model_key: [lang_key, ...]} pour
            le comparatif de langue. Si None, pas de comparatif.
        prompt_type_models: Dict {model_key: [pt_key, ...]} pour
            le comparatif de type de prompt. Si None, pas de comparatif.
        progress_callback: Callback(progress, message).
        selected_gpu: (Optionnel) Dict GPU provenant de detect_all_gpus().
                     Si fourni, utilise ce GPU au lieu de l'auto-détection.

    Returns:
        Dict contenant tous les résultats de benchmark IA.
    """
    # Déterminer les modèles à tester
    if model_keys is None:
        ram_total = psutil.virtual_memory().total / (1024**3)
        compatible = get_compatible_models(ram_total)
        model_keys = list(compatible.keys())

    if not model_keys and not quantization_models and not temperature_models \
       and not language_models and not prompt_type_models:
        return {
            "type": "ai_benchmarks",
            "status": "skipped",
            "reason": "Aucun modèle compatible avec la RAM disponible.",
        }

    # Calculer le nombre total de tâches pour la progression
    n_model_tasks = len(model_keys) if model_keys else 0
    n_quant_tasks = sum(len(qs) for qs in (quantization_models or {}).values())
    n_temp_tasks = len(temperature_models or {})
    n_lang_tasks = len(language_models or {})
    n_pt_tasks = len(prompt_type_models or {})
    total_tasks = n_model_tasks + n_quant_tasks + n_temp_tasks + n_lang_tasks + n_pt_tasks
    if total_tasks == 0:
        total_tasks = 1

    all_results = {}
    start_time = time.time()

    # ── Benchmarks par modèle (classique) ──
    for idx, model_key in enumerate(model_keys or []):
        def model_progress(p, msg, _idx=idx):
            if progress_callback:
                overall = (_idx + p) / total_tasks
                progress_callback(overall, msg)

        if progress_callback:
            progress_callback(
                idx / total_tasks,
                f"Modèle {idx+1}/{n_model_tasks}: "
                f"{AVAILABLE_MODELS.get(model_key, {}).get('name', model_key)}",
            )

        if idx > 0:
            system_cleanup(f"entre modèles ({idx+1}/{n_model_tasks})")

        all_results[model_key] = benchmark_model(
            model_key,
            progress_callback=model_progress,
            selected_gpu=selected_gpu,
        )

    # ── Benchmarks de comparaison de quantification ──
    quant_comparison = {}
    if quantization_models:
        task_offset = n_model_tasks
        for model_key, quant_keys in quantization_models.items():
            n_q = len(quant_keys)

            def quant_progress(p, msg, _offset=task_offset):
                if progress_callback:
                    overall = (_offset + p * n_q) / total_tasks
                    progress_callback(min(overall, 1.0), msg)

            if progress_callback:
                progress_callback(
                    task_offset / total_tasks,
                    f"Comparaison quantification : "
                    f"{AVAILABLE_MODELS.get(model_key, {}).get('name', model_key)}",
                )

            if task_offset > 0:
                system_cleanup(f"avant comparaison quant ({model_key})")

            quant_comparison[model_key] = run_quantization_comparison(
                model_key,
                quant_keys,
                progress_callback=quant_progress,
                selected_gpu=selected_gpu,
            )
            task_offset += n_q

    # ── Benchmarks comparatifs : température ──
    temp_comparison = {}
    if temperature_models:
        task_offset = n_model_tasks + n_quant_tasks
        for model_key, temp_keys in temperature_models.items():
            def temp_progress(p, msg, _offset=task_offset):
                if progress_callback:
                    overall = (_offset + p) / total_tasks
                    progress_callback(min(overall, 1.0), msg)

            if progress_callback:
                progress_callback(
                    task_offset / total_tasks,
                    f"Comparaison température : "
                    f"{AVAILABLE_MODELS.get(model_key, {}).get('name', model_key)}",
                )

            system_cleanup(f"avant comparaison température ({model_key})")

            temp_comparison[model_key] = run_temperature_comparison(
                model_key,
                temp_keys,
                progress_callback=temp_progress,
                selected_gpu=selected_gpu,
            )
            task_offset += 1

    # ── Benchmarks comparatifs : langue ──
    lang_comparison = {}
    if language_models:
        task_offset = n_model_tasks + n_quant_tasks + n_temp_tasks
        for model_key, lang_keys in language_models.items():
            def lang_progress(p, msg, _offset=task_offset):
                if progress_callback:
                    overall = (_offset + p) / total_tasks
                    progress_callback(min(overall, 1.0), msg)

            if progress_callback:
                progress_callback(
                    task_offset / total_tasks,
                    f"Comparaison langues : "
                    f"{AVAILABLE_MODELS.get(model_key, {}).get('name', model_key)}",
                )

            system_cleanup(f"avant comparaison langues ({model_key})")

            lang_comparison[model_key] = run_language_comparison(
                model_key,
                lang_keys,
                progress_callback=lang_progress,
                selected_gpu=selected_gpu,
            )
            task_offset += 1

    # ── Benchmarks comparatifs : type de prompt ──
    pt_comparison = {}
    if prompt_type_models:
        task_offset = n_model_tasks + n_quant_tasks + n_temp_tasks + n_lang_tasks
        for model_key, pt_keys in prompt_type_models.items():
            def pt_progress(p, msg, _offset=task_offset):
                if progress_callback:
                    overall = (_offset + p) / total_tasks
                    progress_callback(min(overall, 1.0), msg)

            if progress_callback:
                progress_callback(
                    task_offset / total_tasks,
                    f"Comparaison types de prompt : "
                    f"{AVAILABLE_MODELS.get(model_key, {}).get('name', model_key)}",
                )

            system_cleanup(f"avant comparaison types de prompt ({model_key})")

            pt_comparison[model_key] = run_prompt_type_comparison(
                model_key,
                pt_keys,
                progress_callback=pt_progress,
                selected_gpu=selected_gpu,
            )
            task_offset += 1

    elapsed_total = time.time() - start_time

    result = {
        "type": "ai_benchmarks",
        "prompt": BENCHMARK_PROMPT,
        "inference_config": INFERENCE_CONFIG,
        "total_time_s": round(elapsed_total, 2),
        "models_tested": len(model_keys or []),
        "results": all_results,
    }

    if quant_comparison:
        result["quantization_comparison"] = quant_comparison
    if temp_comparison:
        result["temperature_comparison"] = temp_comparison
    if lang_comparison:
        result["language_comparison"] = lang_comparison
    if pt_comparison:
        result["prompt_type_comparison"] = pt_comparison

    return result


# =============================================================================
# Benchmark comparatif : Impact de la température
# =============================================================================

def _run_axis_benchmark_runs(
    model,
    n_runs: int,
    prompt: str = BENCHMARK_PROMPT,
    temperature: float = None,
    progress_callback: Optional[Callable] = None,
    progress_base: float = 0.0,
    progress_span: float = 1.0,
    label: str = "",
) -> Dict[str, Any]:
    """
    Exécute n_runs d'inférence avec les paramètres donnés et agrège les résultats.
    Fonction utilitaire partagée par les 3 axes d'analyse.

    Args:
        model: Modèle Llama chargé.
        n_runs: Nombre de runs de benchmark.
        prompt: Prompt à utiliser.
        temperature: Température (None = défaut config).
        progress_callback: Callback(progress, message).
        progress_base: Progression de base (0-1).
        progress_span: Plage de progression pour cette série de runs.
        label: Label pour les messages de progression.

    Returns:
        Dict avec runs individuels + summary agrégé.
    """
    runs = []
    monitor = ResourceMonitor()
    monitor.start()

    for i in range(n_runs):
        if progress_callback:
            p = progress_base + ((i / n_runs) * progress_span)
            progress_callback(p, f"{label} — Run {i+1}/{n_runs}")

        run_result = run_single_inference(
            model,
            prompt=prompt,
            temperature=temperature,
        )
        runs.append(run_result)

    resource_usage = monitor.stop()

    # Agrégation
    successful_runs = [r for r in runs if r["success"]]
    summary = {}
    if successful_runs:
        tps_values = [r["tokens_per_second"] for r in successful_runs]
        ftl_values = [
            r["first_token_latency_s"] for r in successful_runs
            if r["first_token_latency_s"] is not None
        ]
        itl_values = [
            r["avg_inter_token_latency_ms"] for r in successful_runs
            if "avg_inter_token_latency_ms" in r
        ]

        summary = {
            "n_successful_runs": len(successful_runs),
            "n_total_runs": n_runs,
            "avg_tokens_per_second": round(np.mean(tps_values), 2),
            "std_tokens_per_second": round(np.std(tps_values), 2),
            "avg_first_token_latency_s": round(np.mean(ftl_values), 4) if ftl_values else None,
            "avg_inter_token_latency_ms": round(np.mean(itl_values), 2) if itl_values else None,
            "avg_total_time_s": round(
                np.mean([r["total_time_s"] for r in successful_runs]), 4
            ),
            "peak_memory_gb": round(
                max(r["memory_after_gb"] for r in successful_runs), 3
            ),
            "stability": "stable" if len(successful_runs) == n_runs else "unstable",
        }

    return {
        "runs": runs,
        "summary": summary,
        "resource_usage": resource_usage,
    }


def run_temperature_comparison(
    model_key: str,
    temperature_keys: List[str] = None,
    progress_callback: Optional[Callable] = None,
    selected_gpu: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Compare les performances d'un modèle à différentes températures.
    Le modèle est chargé une seule fois puis testé à chaque température.

    Args:
        model_key: Clé du modèle.
        temperature_keys: Liste des clés de température (ex: ["low", "medium", "high"]).
            Si None, toutes les températures sont testées.
        progress_callback: Callback(progress, message).

    Returns:
        Dict structuré avec résultats par température et tableau comparatif.
    """
    if model_key not in AVAILABLE_MODELS:
        return {"error": f"Modèle inconnu : {model_key}"}

    if temperature_keys is None:
        temperature_keys = list(TEMPERATURE_VARIANTS.keys())

    model_config = AVAILABLE_MODELS[model_key]
    n_temps = len(temperature_keys)

    # Nettoyage
    if progress_callback:
        progress_callback(0.01, f"Nettoyage avant test température {model_config['name']}...")
    system_cleanup(f"Temperature {model_config['name']}")

    # Vérifier RAM
    ram_total = psutil.virtual_memory().total / (1024**3)
    if ram_total < model_config["min_ram_gb"]:
        return {
            "model_name": model_config["name"],
            "status": "skipped",
            "reason": f"RAM insuffisante ({ram_total:.1f} Go, {model_config['min_ram_gb']} Go requis)",
        }

    result = {
        "axis": "temperature",
        "model_name": model_config["name"],
        "model_key": model_key,
        "params": model_config["params"],
        "variants_tested": temperature_keys,
        "status": "running",
        "results": {},
    }

    try:
        # Télécharger et charger le modèle une seule fois
        if progress_callback:
            progress_callback(0.05, f"Vérification {model_config['name']}...")
        model_path = download_model(model_key)

        backend_info = detect_best_backend(selected_gpu=selected_gpu)
        result["backend"] = backend_info

        if progress_callback:
            progress_callback(0.10, f"Chargement {model_config['name']}...")
        model = _load_model(model_path, backend_info, n_ctx=INFERENCE_CONFIG["n_ctx"])

        # Échauffement
        if progress_callback:
            progress_callback(0.15, f"Échauffement {model_config['name']}...")
        for _ in range(INFERENCE_CONFIG["n_warmup_runs"]):
            run_single_inference(model, max_tokens=32)

        # Tester chaque température
        n_runs = INFERENCE_CONFIG["n_benchmark_runs"]
        for idx, temp_key in enumerate(temperature_keys):
            temp_config = TEMPERATURE_VARIANTS[temp_key]
            temp_value = temp_config["value"]
            label = f"Température {temp_config['label']}"

            progress_base = 0.20 + (idx / n_temps) * 0.75
            progress_span = 0.75 / n_temps

            axis_result = _run_axis_benchmark_runs(
                model, n_runs,
                temperature=temp_value,
                progress_callback=progress_callback,
                progress_base=progress_base,
                progress_span=progress_span,
                label=label,
            )

            result["results"][temp_key] = {
                "temperature": temp_value,
                "label": temp_config["label"],
                **axis_result,
            }

        result["status"] = "completed"

        # Tableau comparatif
        comparison_table = []
        for tk in temperature_keys:
            tr = result["results"].get(tk, {})
            summary = tr.get("summary", {})
            if summary:
                comparison_table.append({
                    "temperature_key": tk,
                    "temperature": tr.get("temperature", 0),
                    "label": tr.get("label", tk),
                    "tokens_per_second": summary.get("avg_tokens_per_second", 0),
                    "first_token_latency_s": summary.get("avg_first_token_latency_s", 0),
                    "inter_token_latency_ms": summary.get("avg_inter_token_latency_ms", 0),
                    "peak_memory_gb": summary.get("peak_memory_gb", 0),
                    "stability": summary.get("stability", "unknown"),
                })
        result["comparison_table"] = comparison_table

        # Nettoyage
        del model
        gc.collect()

        if progress_callback:
            progress_callback(1.0, f"Test température {model_config['name']} terminé.")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        if progress_callback:
            progress_callback(1.0, f"Erreur : {e}")

    return result


# =============================================================================
# Benchmark comparatif : Impact de la langue
# =============================================================================

def run_language_comparison(
    model_key: str,
    language_keys: List[str] = None,
    progress_callback: Optional[Callable] = None,
    selected_gpu: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Compare les performances d'un modèle selon la langue du prompt.
    Le modèle est chargé une seule fois puis testé avec chaque langue.

    Args:
        model_key: Clé du modèle.
        language_keys: Liste des codes de langue (ex: ["en", "fr", "zh"]).
            Si None, toutes les langues sont testées.
        progress_callback: Callback(progress, message).

    Returns:
        Dict structuré avec résultats par langue et tableau comparatif.
    """
    if model_key not in AVAILABLE_MODELS:
        return {"error": f"Modèle inconnu : {model_key}"}

    if language_keys is None:
        language_keys = list(LANGUAGE_PROMPTS.keys())

    model_config = AVAILABLE_MODELS[model_key]
    n_langs = len(language_keys)

    # Nettoyage
    if progress_callback:
        progress_callback(0.01, f"Nettoyage avant test langues {model_config['name']}...")
    system_cleanup(f"Language {model_config['name']}")

    # Vérifier RAM
    ram_total = psutil.virtual_memory().total / (1024**3)
    if ram_total < model_config["min_ram_gb"]:
        return {
            "model_name": model_config["name"],
            "status": "skipped",
            "reason": f"RAM insuffisante ({ram_total:.1f} Go, {model_config['min_ram_gb']} Go requis)",
        }

    result = {
        "axis": "language",
        "model_name": model_config["name"],
        "model_key": model_key,
        "params": model_config["params"],
        "variants_tested": language_keys,
        "status": "running",
        "results": {},
    }

    try:
        # Charger le modèle une seule fois
        if progress_callback:
            progress_callback(0.05, f"Vérification {model_config['name']}...")
        model_path = download_model(model_key)

        backend_info = detect_best_backend(selected_gpu=selected_gpu)
        result["backend"] = backend_info

        if progress_callback:
            progress_callback(0.10, f"Chargement {model_config['name']}...")
        model = _load_model(model_path, backend_info, n_ctx=INFERENCE_CONFIG["n_ctx"])

        # Échauffement
        if progress_callback:
            progress_callback(0.15, f"Échauffement {model_config['name']}...")
        for _ in range(INFERENCE_CONFIG["n_warmup_runs"]):
            run_single_inference(model, max_tokens=32)

        # Tester chaque langue
        n_runs = INFERENCE_CONFIG["n_benchmark_runs"]
        for idx, lang_key in enumerate(language_keys):
            lang_config = LANGUAGE_PROMPTS[lang_key]
            label = f"{lang_config['flag']} {lang_config['label']}"

            progress_base = 0.20 + (idx / n_langs) * 0.75
            progress_span = 0.75 / n_langs

            axis_result = _run_axis_benchmark_runs(
                model, n_runs,
                prompt=lang_config["prompt"],
                progress_callback=progress_callback,
                progress_base=progress_base,
                progress_span=progress_span,
                label=label,
            )

            result["results"][lang_key] = {
                "language": lang_key,
                "label": lang_config["label"],
                "flag": lang_config["flag"],
                **axis_result,
            }

        result["status"] = "completed"

        # Tableau comparatif
        comparison_table = []
        for lk in language_keys:
            lr = result["results"].get(lk, {})
            summary = lr.get("summary", {})
            if summary:
                comparison_table.append({
                    "language_key": lk,
                    "label": lr.get("label", lk),
                    "flag": lr.get("flag", ""),
                    "tokens_per_second": summary.get("avg_tokens_per_second", 0),
                    "first_token_latency_s": summary.get("avg_first_token_latency_s", 0),
                    "inter_token_latency_ms": summary.get("avg_inter_token_latency_ms", 0),
                    "peak_memory_gb": summary.get("peak_memory_gb", 0),
                    "stability": summary.get("stability", "unknown"),
                })
        result["comparison_table"] = comparison_table

        # Nettoyage
        del model
        gc.collect()

        if progress_callback:
            progress_callback(1.0, f"Test langues {model_config['name']} terminé.")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        if progress_callback:
            progress_callback(1.0, f"Erreur : {e}")

    return result


# =============================================================================
# Benchmark comparatif : Impact du type de prompt
# =============================================================================

def run_prompt_type_comparison(
    model_key: str,
    prompt_type_keys: List[str] = None,
    progress_callback: Optional[Callable] = None,
    selected_gpu: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Compare les performances d'un modèle selon le type de prompt.
    Le modèle est chargé une seule fois puis testé avec chaque type.

    Args:
        model_key: Clé du modèle.
        prompt_type_keys: Liste des clés de type (ex: ["code", "reasoning"]).
            Si None, tous les types sont testés.
        progress_callback: Callback(progress, message).

    Returns:
        Dict structuré avec résultats par type de prompt et tableau comparatif.
    """
    if model_key not in AVAILABLE_MODELS:
        return {"error": f"Modèle inconnu : {model_key}"}

    if prompt_type_keys is None:
        prompt_type_keys = list(PROMPT_TYPE_VARIANTS.keys())

    model_config = AVAILABLE_MODELS[model_key]
    n_types = len(prompt_type_keys)

    # Nettoyage
    if progress_callback:
        progress_callback(0.01, f"Nettoyage avant test types de prompt {model_config['name']}...")
    system_cleanup(f"PromptType {model_config['name']}")

    # Vérifier RAM
    ram_total = psutil.virtual_memory().total / (1024**3)
    if ram_total < model_config["min_ram_gb"]:
        return {
            "model_name": model_config["name"],
            "status": "skipped",
            "reason": f"RAM insuffisante ({ram_total:.1f} Go, {model_config['min_ram_gb']} Go requis)",
        }

    result = {
        "axis": "prompt_type",
        "model_name": model_config["name"],
        "model_key": model_key,
        "params": model_config["params"],
        "variants_tested": prompt_type_keys,
        "status": "running",
        "results": {},
    }

    try:
        # Charger le modèle une seule fois
        if progress_callback:
            progress_callback(0.05, f"Vérification {model_config['name']}...")
        model_path = download_model(model_key)

        backend_info = detect_best_backend(selected_gpu=selected_gpu)
        result["backend"] = backend_info

        if progress_callback:
            progress_callback(0.10, f"Chargement {model_config['name']}...")
        model = _load_model(model_path, backend_info, n_ctx=INFERENCE_CONFIG["n_ctx"])

        # Échauffement
        if progress_callback:
            progress_callback(0.15, f"Échauffement {model_config['name']}...")
        for _ in range(INFERENCE_CONFIG["n_warmup_runs"]):
            run_single_inference(model, max_tokens=32)

        # Tester chaque type de prompt
        n_runs = INFERENCE_CONFIG["n_benchmark_runs"]
        for idx, pt_key in enumerate(prompt_type_keys):
            pt_config = PROMPT_TYPE_VARIANTS[pt_key]
            label = f"{pt_config['icon']} {pt_config['label']}"

            progress_base = 0.20 + (idx / n_types) * 0.75
            progress_span = 0.75 / n_types

            axis_result = _run_axis_benchmark_runs(
                model, n_runs,
                prompt=pt_config["prompt"],
                progress_callback=progress_callback,
                progress_base=progress_base,
                progress_span=progress_span,
                label=label,
            )

            result["results"][pt_key] = {
                "prompt_type": pt_key,
                "label": pt_config["label"],
                "icon": pt_config["icon"],
                **axis_result,
            }

        result["status"] = "completed"

        # Tableau comparatif
        comparison_table = []
        for ptk in prompt_type_keys:
            ptr = result["results"].get(ptk, {})
            summary = ptr.get("summary", {})
            if summary:
                comparison_table.append({
                    "prompt_type_key": ptk,
                    "label": ptr.get("label", ptk),
                    "icon": ptr.get("icon", ""),
                    "tokens_per_second": summary.get("avg_tokens_per_second", 0),
                    "first_token_latency_s": summary.get("avg_first_token_latency_s", 0),
                    "inter_token_latency_ms": summary.get("avg_inter_token_latency_ms", 0),
                    "peak_memory_gb": summary.get("peak_memory_gb", 0),
                    "stability": summary.get("stability", "unknown"),
                })
        result["comparison_table"] = comparison_table

        # Nettoyage
        del model
        gc.collect()

        if progress_callback:
            progress_callback(1.0, f"Test types de prompt {model_config['name']} terminé.")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        if progress_callback:
            progress_callback(1.0, f"Erreur : {e}")

    return result


if __name__ == "__main__":
    def print_progress(p, msg):
        bar = "█" * int(p * 30) + "░" * (30 - int(p * 30))
        print(f"\r  [{bar}] {p*100:.0f}% - {msg}", end="", flush=True)

    print("Lancement des benchmarks IA...\n")
    # Tester uniquement le plus petit modèle
    results = run_all_ai_benchmarks(
        model_keys=["tinyllama-1.1b"],
        progress_callback=print_progress
    )
    print("\n\nTerminé !")
    
    import json
    print(json.dumps(results, indent=2, default=str))
