"""
ComputeLLM - Module de gestion des résultats.
Sauvegarde, chargement, comparaison et export des résultats de benchmark.
"""

import json
import csv
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.config import RESULTS_DIR


def generate_result_id() -> str:
    """Génère un identifiant unique pour un résultat de benchmark."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_results(
    hardware_info: Dict[str, Any],
    classic_results: Optional[Dict[str, Any]],
    ai_results: Optional[Dict[str, Any]],
    result_id: str = None,
) -> Path:
    """
    Sauvegarde tous les résultats de benchmark dans un fichier JSON.
    
    Args:
        hardware_info: Informations matérielles.
        classic_results: Résultats des benchmarks classiques.
        ai_results: Résultats des benchmarks IA.
        result_id: Identifiant unique (généré automatiquement si None).
    
    Returns:
        Path du fichier sauvegardé.
    """
    if result_id is None:
        result_id = generate_result_id()

    # Construire le document complet
    document = {
        "id": result_id,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "hardware": hardware_info,
    }

    if classic_results:
        document["classic_benchmarks"] = classic_results

    if ai_results:
        document["ai_benchmarks"] = ai_results

    # Construire un nom de fichier descriptif
    machine_name = _get_machine_name(hardware_info)
    filename = f"benchmark_{machine_name}_{result_id}.json"
    filepath = RESULTS_DIR / filename

    # Sauvegarder en JSON
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2, ensure_ascii=False, default=str)

    return filepath


def _get_machine_name(hardware_info: Dict[str, Any]) -> str:
    """Génère un nom court pour la machine à partir des infos matérielles."""
    parts = []

    # CPU
    cpu_model = hardware_info.get("cpu", {}).get("model", "")
    if "Apple" in cpu_model or hardware_info.get("cpu", {}).get("is_apple_silicon"):
        parts.append("Apple")
        chip = hardware_info.get("cpu", {}).get("chip", "")
        if chip:
            parts.append(chip.replace(" ", ""))
        else:
            # Essayer d'extraire de la chaîne model
            for m in ["M1", "M2", "M3", "M4"]:
                if m in cpu_model:
                    parts.append(m)
                    break
    else:
        if "Intel" in cpu_model:
            parts.append("Intel")
        elif "AMD" in cpu_model:
            parts.append("AMD")
        else:
            parts.append("CPU")

    # GPU — prendre le GPU sélectionné pour le benchmark, ou le premier
    gpus = hardware_info.get("gpu", {}).get("gpus", [])
    selected_gpu = hardware_info.get("selected_gpu", {})
    if selected_gpu:
        # Retrouver le GPU sélectionné dans la liste
        sel_idx = selected_gpu.get("gpu_index", 0)
        target_gpu = next((g for g in gpus if g.get("gpu_index") == sel_idx), gpus[0] if gpus else None)
    elif gpus:
        target_gpu = gpus[0]
    else:
        target_gpu = None

    if target_gpu:
        gpu_name = target_gpu.get("name", "")
        gpu_type = target_gpu.get("type", "")
        if "NVIDIA" in gpu_name or gpu_type == "NVIDIA":
            # Extraire le nom court du GPU
            for keyword in ["RTX", "GTX", "A100", "A6000", "H100", "V100"]:
                if keyword in gpu_name:
                    parts.append(keyword)
                    break
        elif gpu_type == "AMD" or "AMD" in gpu_name or "Radeon" in gpu_name:
            # Extraire le nom court du GPU AMD
            for keyword in ["RX", "Instinct", "Radeon", "W7", "W6", "FirePro"]:
                if keyword.lower() in gpu_name.lower():
                    parts.append(keyword)
                    break
            else:
                parts.append("AMDGPU")
        elif gpu_type == "Intel" or "Intel" in gpu_name:
            # Extraire le nom court du GPU Intel
            for keyword in ["Arc", "DG1", "DG2", "Flex", "A770", "A750", "A380"]:
                if keyword.lower() in gpu_name.lower():
                    parts.append(keyword)
                    break
            else:
                parts.append("IntelGPU")
        elif "Apple" in str(gpu_type):
            if "Apple" not in parts:
                parts.append("Metal")

    # OS
    os_sys = hardware_info.get("os", {}).get("system", "")
    if os_sys == "Darwin":
        parts.append("macOS")
    elif os_sys == "Windows":
        parts.append("Win")
    elif os_sys == "Linux":
        parts.append("Linux")

    name = "_".join(parts) if parts else "unknown"
    # Nettoyer le nom
    name = "".join(c if c.isalnum() or c == "_" else "" for c in name)
    return name


def load_result(filepath: str) -> Dict[str, Any]:
    """Charge un résultat de benchmark depuis un fichier JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def list_results() -> List[Dict[str, Any]]:
    """
    Liste tous les résultats de benchmark sauvegardés.
    
    Returns:
        Liste de dicts avec les métadonnées de chaque résultat.
    """
    results = []
    if not RESULTS_DIR.exists():
        return results

    for filepath in sorted(RESULTS_DIR.glob("benchmark_*.json"), reverse=True):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extraire les métadonnées
            meta = {
                "filepath": str(filepath),
                "filename": filepath.name,
                "id": data.get("id", ""),
                "timestamp": data.get("timestamp", ""),
                "machine": _get_machine_name(data.get("hardware", {})),
            }

            # Résumé CPU
            cpu = data.get("hardware", {}).get("cpu", {})
            meta["cpu"] = cpu.get("model", "Unknown")

            # Résumé GPU — liste tous les GPUs et indique le GPU sélectionné
            gpus = data.get("hardware", {}).get("gpu", {}).get("gpus", [])
            selected_gpu = data.get("hardware", {}).get("selected_gpu", {})
            if selected_gpu:
                meta["gpu"] = selected_gpu.get("name", gpus[0]["name"] if gpus else "None")
                meta["gpu_index"] = selected_gpu.get("gpu_index", 0)
            elif gpus:
                meta["gpu"] = gpus[0]["name"]
            else:
                meta["gpu"] = "None"
            meta["gpu_count"] = len(gpus)

            # Backend
            meta["backend"] = data.get("hardware", {}).get("gpu", {}).get("primary_backend", "cpu")

            # RAM
            meta["ram_gb"] = data.get("hardware", {}).get("ram", {}).get("total_gb", 0)

            # Résultat clé IA (tokens/s du modèle le plus gros testé)
            ai_results = data.get("ai_benchmarks", {}).get("results", {})
            if ai_results:
                best_tps = 0
                best_model = ""
                for model_key, model_result in ai_results.items():
                    summary = model_result.get("summary", {})
                    tps = summary.get("avg_tokens_per_second", 0)
                    if tps > 0:
                        best_tps = tps
                        best_model = model_result.get("model", model_key)
                meta["best_tokens_per_second"] = best_tps
                meta["best_model"] = best_model

            # Résumé quantification
            quant_comp = data.get("ai_benchmarks", {}).get("quantization_comparison", {})
            if quant_comp:
                meta["has_quantization_comparison"] = True
                quant_models = []
                for mk, comp_data in quant_comp.items():
                    n_variants = len(comp_data.get("comparison_table", []))
                    quant_models.append(f"{comp_data.get('model_name', mk)} ({n_variants} quants)")
                meta["quantization_models"] = ", ".join(quant_models)
            else:
                meta["has_quantization_comparison"] = False

            results.append(meta)
        except Exception:
            continue

    return results


def compare_results(result_files: List[str]) -> Dict[str, Any]:
    """
    Compare les résultats de plusieurs machines.
    
    Args:
        result_files: Liste des chemins vers les fichiers de résultats.
    
    Returns:
        Dict structuré pour la comparaison.
    """
    machines = []

    for filepath in result_files:
        try:
            data = load_result(filepath)
            machine = {
                "name": _get_machine_name(data.get("hardware", {})),
                "id": data.get("id", ""),
                "hardware": data.get("hardware", {}),
            }

            # Extraire les résultats clés des benchmarks classiques
            classic = data.get("classic_benchmarks", {}).get("benchmarks", {})

            # CPU Multi-Thread (plus grande taille)
            cpu_mt = classic.get("cpu_multi_thread", {}).get("results", {})
            if cpu_mt:
                largest = list(cpu_mt.values())[-1]  # Dernière = plus grande
                machine["cpu_mt_gflops"] = largest.get("gflops", 0)

            # CPU Single-Thread
            cpu_st = classic.get("cpu_single_thread", {}).get("results", {})
            if cpu_st:
                largest = list(cpu_st.values())[-1]
                machine["cpu_st_gflops"] = largest.get("gflops", 0)

            # Mémoire
            mem = classic.get("memory_bandwidth", {}).get("results", {})
            if mem:
                machine["mem_read_gb_s"] = mem.get("read", {}).get("bandwidth_gb_s", 0)
                machine["mem_write_gb_s"] = mem.get("write", {}).get("bandwidth_gb_s", 0)
                machine["mem_copy_gb_s"] = mem.get("copy", {}).get("bandwidth_gb_s", 0)

            # GPU
            gpu = classic.get("gpu_compute", {})
            if gpu.get("status") == "completed":
                gpu_results = gpu.get("results", {})
                if gpu_results:
                    largest = list(gpu_results.values())[-1]
                    machine["gpu_gflops"] = largest.get("gflops", 0)
                    machine["gpu_backend"] = gpu.get("backend", "")

            # IA
            ai_results = data.get("ai_benchmarks", {}).get("results", {})
            machine["ai_models"] = {}
            for model_key, model_data in ai_results.items():
                summary = model_data.get("summary", {})
                if summary:
                    machine["ai_models"][model_key] = {
                        "model_name": model_data.get("model", model_key),
                        "tokens_per_second": summary.get("avg_tokens_per_second", 0),
                        "first_token_latency_s": summary.get("avg_first_token_latency_s", 0),
                        "peak_memory_gb": summary.get("peak_memory_gb", 0),
                        "stability": summary.get("stability", "unknown"),
                    }

            machines.append(machine)
        except Exception as e:
            continue

    return {
        "n_machines": len(machines),
        "machines": machines,
    }


def export_to_csv(result_file: str, output_path: str = None) -> Path:
    """
    Exporte un résultat de benchmark en CSV.
    
    Args:
        result_file: Chemin vers le fichier JSON.
        output_path: Chemin de sortie CSV (auto-généré si None).
    
    Returns:
        Path du fichier CSV.
    """
    data = load_result(result_file)

    if output_path is None:
        output_path = Path(result_file).with_suffix(".csv")
    else:
        output_path = Path(output_path)

    rows = []

    # Ligne d'info machine
    hw = data.get("hardware", {})
    machine_name = _get_machine_name(hw)

    # Benchmarks classiques
    classic = data.get("classic_benchmarks", {}).get("benchmarks", {})

    for bench_name, bench_data in classic.items():
        if isinstance(bench_data, dict) and "results" in bench_data:
            for test_name, test_data in bench_data["results"].items():
                row = {
                    "machine": machine_name,
                    "benchmark_type": "classic",
                    "test": bench_data.get("test", bench_name),
                    "sub_test": test_name,
                    "metric": "gflops" if "gflops" in test_data else "bandwidth_gb_s",
                    "value": test_data.get("gflops", test_data.get("bandwidth_gb_s", 0)),
                    "mean_time_s": test_data.get("mean_s", test_data.get("mean_s", 0)),
                }
                rows.append(row)

    # Benchmarks IA
    ai_results = data.get("ai_benchmarks", {}).get("results", {})
    for model_key, model_data in ai_results.items():
        summary = model_data.get("summary", {})
        if summary:
            rows.append({
                "machine": machine_name,
                "benchmark_type": "ai_inference",
                "test": model_data.get("model", model_key),
                "sub_test": "tokens_per_second",
                "metric": "tokens/s",
                "value": summary.get("avg_tokens_per_second", 0),
                "mean_time_s": summary.get("avg_total_time_s", 0),
            })
            rows.append({
                "machine": machine_name,
                "benchmark_type": "ai_inference",
                "test": model_data.get("model", model_key),
                "sub_test": "first_token_latency",
                "metric": "seconds",
                "value": summary.get("avg_first_token_latency_s", 0),
                "mean_time_s": 0,
            })
        elif model_data.get("status") in ("error", "skipped"):
            rows.append({
                "machine": machine_name,
                "benchmark_type": "ai_inference",
                "test": model_data.get("model", model_key),
                "sub_test": model_data.get("status", "error"),
                "metric": "error",
                "value": 0,
                "mean_time_s": 0,
                "error": model_data.get("error", model_data.get("reason", "")),
            })

    # Benchmarks de comparaison de quantification
    quant_comp = data.get("ai_benchmarks", {}).get("quantization_comparison", {})
    for model_key, comp_data in quant_comp.items():
        model_name = comp_data.get("model_name", model_key)
        for qk, qr in comp_data.get("results", {}).items():
            summary = qr.get("summary", {})
            if summary:
                rows.append({
                    "machine": machine_name,
                    "benchmark_type": "quantization_comparison",
                    "test": f"{model_name} {qk}",
                    "sub_test": "tokens_per_second",
                    "metric": "tokens/s",
                    "value": summary.get("avg_tokens_per_second", 0),
                    "mean_time_s": summary.get("avg_total_time_s", 0),
                })
                rows.append({
                    "machine": machine_name,
                    "benchmark_type": "quantization_comparison",
                    "test": f"{model_name} {qk}",
                    "sub_test": "first_token_latency",
                    "metric": "seconds",
                    "value": summary.get("avg_first_token_latency_s", 0),
                    "mean_time_s": 0,
                })
                rows.append({
                    "machine": machine_name,
                    "benchmark_type": "quantization_comparison",
                    "test": f"{model_name} {qk}",
                    "sub_test": "peak_memory",
                    "metric": "gb",
                    "value": summary.get("peak_memory_gb", 0),
                    "mean_time_s": 0,
                })
                rows.append({
                    "machine": machine_name,
                    "benchmark_type": "quantization_comparison",
                    "test": f"{model_name} {qk}",
                    "sub_test": "model_load_time",
                    "metric": "seconds",
                    "value": qr.get("model_load_time_s", 0),
                    "mean_time_s": 0,
                })
                rows.append({
                    "machine": machine_name,
                    "benchmark_type": "quantization_comparison",
                    "test": f"{model_name} {qk}",
                    "sub_test": "file_size",
                    "metric": "gb",
                    "value": qr.get("actual_file_size_gb", qr.get("file_size_gb", 0)),
                    "mean_time_s": 0,
                })

    # Écrire le CSV
    if rows:
        # Collecter tous les noms de colonnes (certaines lignes ont 'error')
        all_keys = set()
        for r in rows:
            all_keys.update(r.keys())
        fieldnames = sorted(all_keys, key=lambda k: list(rows[0].keys()).index(k) if k in rows[0] else 999)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return output_path


if __name__ == "__main__":
    # Test: lister les résultats existants
    results = list_results()
    if results:
        print(f"Résultats trouvés : {len(results)}")
        for r in results:
            print(f"  - {r['filename']} ({r['machine']}) - {r.get('best_tokens_per_second', 'N/A')} tok/s")
    else:
        print("Aucun résultat de benchmark trouvé.")
