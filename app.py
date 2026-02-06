"""
ComputeLLM - Application Streamlit (Interface utilisateur).
Interface graphique principale avec trois pages :
  1. Matériel - Détection et affichage des informations matérielles
  2. Benchmark - Lancement des benchmarks (bouton unique)
  3. Résultats - Affichage, comparaison et visualisation
"""

import streamlit as st
import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent))

from src.hardware_detect import get_full_hardware_info, get_hardware_summary
from src.benchmark_classic import run_all_classic_benchmarks
from src.benchmark_ai import (
    run_all_ai_benchmarks,
    list_available_models,
    get_compatible_models,
    is_model_downloaded,
    download_model,
    detect_best_backend,
)
from src.results_manager import (
    save_results,
    list_results,
    load_result,
    compare_results,
    export_to_csv,
)
from src.config import AVAILABLE_MODELS, RESULTS_DIR

# =============================================================================
# Configuration de la page Streamlit
# =============================================================================
st.set_page_config(
    page_title="ComputeLLM - AI Hardware Benchmark",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CSS personnalisé
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 0.9rem;
        opacity: 0.8;
    }
    .metric-card h2 {
        margin: 0.5rem 0 0 0;
        font-size: 1.8rem;
    }
    .status-ok { color: #4CAF50; font-weight: bold; }
    .status-warn { color: #FF9800; font-weight: bold; }
    .status-error { color: #f44336; font-weight: bold; }
    .benchmark-btn {
        display: block;
        margin: 2rem auto;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State Initialization
# =============================================================================
if "hardware_info" not in st.session_state:
    st.session_state.hardware_info = None
if "benchmark_running" not in st.session_state:
    st.session_state.benchmark_running = False
if "classic_results" not in st.session_state:
    st.session_state.classic_results = None
if "ai_results" not in st.session_state:
    st.session_state.ai_results = None
if "last_save_path" not in st.session_state:
    st.session_state.last_save_path = None


# =============================================================================
# Sidebar Navigation
# =============================================================================
st.sidebar.markdown("## 🖥️ ComputeLLM")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Matériel", "🚀 Benchmark", "📊 Résultats"],
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Version** 1.0.0")
st.sidebar.markdown("Benchmark Hardware IA")
st.sidebar.markdown("Multiplateforme (macOS / Windows)")


# =============================================================================
# PAGE 1 : Détection Matérielle
# =============================================================================
def page_hardware():
    st.markdown('<h1 class="main-header">🖥️ Détection Matérielle</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyse automatique de votre configuration</p>', unsafe_allow_html=True)

    # Bouton de détection
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Détecter le matériel", use_container_width=True, type="primary"):
            with st.spinner("Analyse du matériel en cours..."):
                st.session_state.hardware_info = get_full_hardware_info()

    if st.session_state.hardware_info is None:
        st.info("Cliquez sur le bouton ci-dessus pour détecter votre matériel.")
        return

    hw = st.session_state.hardware_info

    # --- Système d'exploitation ---
    st.markdown("### 💻 Système d'exploitation")
    os_info = hw["os"]
    cols = st.columns(4)
    cols[0].metric("OS", os_info["system"])
    cols[1].metric("Version", os_info["release"])
    cols[2].metric("Architecture", os_info["architecture"])
    cols[3].metric("Python", os_info["python_version"])

    st.markdown("---")

    # --- CPU ---
    st.markdown("### ⚙️ Processeur (CPU)")
    cpu = hw["cpu"]
    cols = st.columns(4)
    cols[0].metric("Modèle", cpu.get("model", "Unknown"))
    cols[1].metric("Cœurs physiques", cpu.get("physical_cores", "?"))
    cols[2].metric("Cœurs logiques", cpu.get("logical_cores", "?"))
    arch_type = cpu.get("architecture_type", cpu.get("architecture", "?"))
    cols[3].metric("Architecture", arch_type)

    if cpu.get("is_apple_silicon"):
        cols2 = st.columns(4)
        cols2[0].metric("Type", "Apple Silicon ✅")
        perf = cpu.get("performance_cores", "?")
        eff = cpu.get("efficiency_cores", "?")
        cols2[1].metric("Cœurs Performance", perf)
        cols2[2].metric("Cœurs Efficience", eff)
        if cpu.get("frequency_mhz"):
            cols2[3].metric("Fréquence", f"{cpu['frequency_mhz'].get('current', '?')} MHz")

    st.markdown("---")

    # --- Mémoire RAM ---
    st.markdown("### 🧠 Mémoire RAM")
    ram = hw["ram"]
    cols = st.columns(4)
    cols[0].metric("Total", f"{ram['total_gb']} Go")
    cols[1].metric("Disponible", f"{ram['available_gb']} Go")
    cols[2].metric("Utilisée", f"{ram['percent_used']}%")
    mem_type = "Unifiée (CPU+GPU)" if ram.get("unified_memory") else "Dédiée"
    cols[3].metric("Type", mem_type)

    if ram.get("swap_total_gb", 0) > 0:
        st.caption(f"Swap : {ram['swap_used_gb']:.1f} / {ram['swap_total_gb']:.1f} Go")

    st.markdown("---")

    # --- GPU ---
    st.markdown("### 🎮 GPU & Accélération")
    gpu = hw["gpu"]

    if gpu["gpus"]:
        for g in gpu["gpus"]:
            cols = st.columns(4)
            cols[0].metric("GPU", g["name"])
            cols[1].metric("Type", g["type"])
            cols[2].metric("Backend", g["backend"].upper())

            if "vram_total_mb" in g:
                cols[3].metric("VRAM", f"{g['vram_total_mb']:.0f} Mo")
            elif "unified_memory_gb" in g:
                cols[3].metric("Mémoire", f"{g['unified_memory_gb']} Go (unifiée)")
            elif "vram" in g:
                cols[3].metric("VRAM", g["vram"])
    else:
        st.warning("Aucun GPU détecté. L'inférence utilisera le CPU.")

    cols_backend = st.columns(3)
    cols_backend[0].metric("Backend principal", gpu["primary_backend"].upper())
    cols_backend[1].metric("Backends disponibles", ", ".join(b.upper() for b in gpu["backends"]))

    # Bibliothèques Python
    py_backends = gpu.get("python_backends", {})
    if py_backends:
        st.markdown("**Bibliothèques Python détectées :**")
        py_cols = st.columns(3)
        if py_backends.get("llama_cpp"):
            py_cols[0].success(f"✅ llama-cpp-python {py_backends.get('llama_cpp_version', '')}")
        else:
            py_cols[0].error("❌ llama-cpp-python non installé")

        if py_backends.get("pytorch"):
            ver = py_backends.get("pytorch_version", "")
            cuda_str = f" (CUDA {py_backends['pytorch_cuda_version']})" if py_backends.get("pytorch_cuda") else ""
            mps_str = " (MPS ✅)" if py_backends.get("pytorch_mps") else ""
            py_cols[1].success(f"✅ PyTorch {ver}{cuda_str}{mps_str}")
        else:
            py_cols[1].warning("⚠️ PyTorch non installé (GPU benchmark indisponible)")

    # Modèles compatibles
    st.markdown("---")
    st.markdown("### 📦 Modèles LLM compatibles")
    ram_total = ram["total_gb"]
    compatible = get_compatible_models(ram_total)

    for key, model in AVAILABLE_MODELS.items():
        is_compat = key in compatible
        is_downloaded = is_model_downloaded(key)
        icon = "✅" if is_compat else "❌"
        dl_icon = "📥" if is_downloaded else "⬜"

        cols = st.columns([0.5, 2, 1, 1, 1, 1])
        cols[0].write(icon)
        cols[1].write(f"**{model['name']}** ({model['params']})")
        cols[2].write(f"{model['size_gb']} Go")
        cols[3].write(f"RAM min: {model['min_ram_gb']} Go")
        cols[4].write(dl_icon + (" Téléchargé" if is_downloaded else " Non téléchargé"))
        cols[5].write("Compatible" if is_compat else "Incompatible")

    # Export JSON brut
    with st.expander("📋 Données brutes (JSON)"):
        st.json(hw)


# =============================================================================
# PAGE 2 : Benchmark
# =============================================================================
def page_benchmark():
    st.markdown('<h1 class="main-header">🚀 Benchmark</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Lancer l\'ensemble des benchmarks matériels et IA</p>', unsafe_allow_html=True)

    # Vérifier que le matériel a été détecté
    if st.session_state.hardware_info is None:
        st.warning("⚠️ Veuillez d'abord détecter le matériel dans la page 'Matériel'.")
        if st.button("🔍 Détecter le matériel maintenant"):
            with st.spinner("Détection..."):
                st.session_state.hardware_info = get_full_hardware_info()
            st.rerun()
        return

    hw = st.session_state.hardware_info

    # Configuration des benchmarks
    st.markdown("### ⚙️ Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Benchmarks classiques**")
        run_classic = st.checkbox("CPU Single-Thread", value=True)
        run_classic_mt = st.checkbox("CPU Multi-Thread", value=True)
        run_memory = st.checkbox("Bande passante mémoire", value=True)
        run_gpu = st.checkbox("GPU (si disponible)", value=True)

    with col2:
        st.markdown("**Benchmarks IA (Inférence LLM)**")
        ram_total = hw["ram"]["total_gb"]
        compatible = get_compatible_models(ram_total)

        selected_models = []
        for key, model in compatible.items():
            default = key == "tinyllama-1.1b"  # Sélectionner le petit par défaut
            if st.checkbox(
                f"{model['name']} ({model['params']}) - {model['size_gb']} Go",
                value=default,
                key=f"model_{key}",
            ):
                selected_models.append(key)

        if not compatible:
            st.warning("Aucun modèle compatible avec votre RAM.")

    st.markdown("---")

    # Résumé de la configuration
    backend_info = detect_best_backend()
    st.markdown("### 📋 Résumé")
    cols = st.columns(4)
    cols[0].metric("Backend IA", backend_info["backend"].upper())
    cols[1].metric("Modèles sélectionnés", len(selected_models))
    cols[2].metric("RAM disponible", f"{hw['ram']['available_gb']} Go")
    n_tests = sum([run_classic, run_classic_mt, run_memory, run_gpu]) + len(selected_models)
    cols[3].metric("Tests à exécuter", n_tests)

    st.markdown("---")

    # ===== BOUTON UNIQUE DE LANCEMENT =====
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        launch = st.button(
            "🚀 LANCER TOUS LES BENCHMARKS",
            use_container_width=True,
            type="primary",
            disabled=st.session_state.benchmark_running,
        )

    if launch:
        st.session_state.benchmark_running = True
        st.session_state.classic_results = None
        st.session_state.ai_results = None

        total_start = time.time()

        # ============================
        # BENCHMARKS CLASSIQUES
        # ============================
        if any([run_classic, run_classic_mt, run_memory, run_gpu]):
            st.markdown("### 📊 Benchmarks Classiques")
            classic_progress = st.progress(0.0)
            classic_status = st.status("Benchmarks classiques en cours...", expanded=True)

            def classic_callback(p, msg):
                classic_progress.progress(min(p, 1.0))
                classic_status.update(label=msg)

            with classic_status:
                st.write("Exécution des tests CPU, GPU et mémoire...")
                try:
                    classic_results = run_all_classic_benchmarks(
                        progress_callback=classic_callback
                    )
                    st.session_state.classic_results = classic_results
                    classic_status.update(
                        label=f"✅ Benchmarks classiques terminés ({classic_results['total_time_s']:.1f}s)",
                        state="complete"
                    )
                except Exception as e:
                    classic_status.update(label=f"❌ Erreur : {e}", state="error")
                    st.error(f"Erreur benchmarks classiques : {e}")

            classic_progress.progress(1.0)
        else:
            st.info("Benchmarks classiques désactivés.")

        # ============================
        # BENCHMARKS IA
        # ============================
        if selected_models:
            st.markdown("### 🤖 Benchmarks IA (Inférence LLM)")
            ai_progress = st.progress(0.0)
            ai_status = st.status("Benchmarks IA en cours...", expanded=True)

            def ai_callback(p, msg):
                ai_progress.progress(min(p, 1.0))
                ai_status.update(label=msg)

            with ai_status:
                # Téléchargement des modèles si nécessaire
                for model_key in selected_models:
                    if not is_model_downloaded(model_key):
                        model_info = AVAILABLE_MODELS[model_key]
                        st.write(f"📥 Téléchargement de {model_info['name']}...")
                        try:
                            download_model(model_key)
                            st.write(f"✅ {model_info['name']} téléchargé.")
                        except Exception as e:
                            st.error(f"❌ Erreur téléchargement {model_info['name']}: {e}")

                # Exécution des benchmarks
                st.write("Exécution des inférences...")
                try:
                    ai_results = run_all_ai_benchmarks(
                        model_keys=selected_models,
                        progress_callback=ai_callback,
                    )
                    st.session_state.ai_results = ai_results
                    ai_status.update(
                        label=f"✅ Benchmarks IA terminés ({ai_results['total_time_s']:.1f}s)",
                        state="complete"
                    )
                except Exception as e:
                    ai_status.update(label=f"❌ Erreur : {e}", state="error")
                    st.error(f"Erreur benchmarks IA : {e}")

            ai_progress.progress(1.0)
        else:
            st.info("Aucun modèle IA sélectionné.")

        # ============================
        # SAUVEGARDE AUTOMATIQUE
        # ============================
        total_time = time.time() - total_start

        st.markdown("---")
        st.markdown("### 💾 Sauvegarde")

        try:
            save_path = save_results(
                hardware_info=hw,
                classic_results=st.session_state.classic_results,
                ai_results=st.session_state.ai_results,
            )
            st.session_state.last_save_path = str(save_path)
            st.success(f"✅ Résultats sauvegardés : `{save_path.name}`")
        except Exception as e:
            st.error(f"❌ Erreur sauvegarde : {e}")

        st.markdown(f"**Temps total : {total_time:.1f} secondes**")

        st.session_state.benchmark_running = False
        st.balloons()

    # Afficher un résumé rapide si des résultats sont en session
    if st.session_state.classic_results or st.session_state.ai_results:
        st.markdown("---")
        st.markdown("### 📋 Derniers résultats")

        if st.session_state.classic_results:
            benchmarks = st.session_state.classic_results.get("benchmarks", {})

            cols = st.columns(4)

            # CPU ST
            cpu_st = benchmarks.get("cpu_single_thread", {}).get("results", {})
            if cpu_st:
                largest = list(cpu_st.values())[-1]
                cols[0].metric("CPU Single-Thread", f"{largest.get('gflops', 0)} GFLOPS")

            # CPU MT
            cpu_mt = benchmarks.get("cpu_multi_thread", {}).get("results", {})
            if cpu_mt:
                largest = list(cpu_mt.values())[-1]
                cols[1].metric("CPU Multi-Thread", f"{largest.get('gflops', 0)} GFLOPS")

            # Mémoire
            mem = benchmarks.get("memory_bandwidth", {}).get("results", {})
            if mem:
                read_bw = mem.get("read", {}).get("bandwidth_gb_s", 0)
                cols[2].metric("Mémoire (lecture)", f"{read_bw} Go/s")

            # GPU
            gpu = benchmarks.get("gpu_compute", {})
            if gpu.get("status") == "completed":
                gpu_res = gpu.get("results", {})
                if gpu_res:
                    largest = list(gpu_res.values())[-1]
                    cols[3].metric("GPU", f"{largest.get('gflops', 0)} GFLOPS")
            else:
                cols[3].metric("GPU", gpu.get("reason", "N/A"))

        if st.session_state.ai_results:
            ai_res = st.session_state.ai_results.get("results", {})
            if ai_res:
                cols = st.columns(len(ai_res))
                for i, (key, data) in enumerate(ai_res.items()):
                    summary = data.get("summary", {})
                    if summary:
                        tps = summary.get("avg_tokens_per_second", 0)
                        ftl = summary.get("avg_first_token_latency_s", 0)
                        cols[i].metric(
                            data.get("model", key),
                            f"{tps} tok/s",
                            f"Latence 1er token : {ftl:.3f}s"
                        )
                    elif data.get("status") == "skipped":
                        cols[i].metric(
                            data.get("model", key),
                            "Ignoré",
                            data.get("reason", "")
                        )

        st.info("📊 Consultez la page **Résultats** pour une analyse détaillée.")


# =============================================================================
# PAGE 3 : Résultats
# =============================================================================
def page_results():
    st.markdown('<h1 class="main-header">📊 Résultats</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Visualisation et comparaison des benchmarks</p>', unsafe_allow_html=True)

    # Charger la liste des résultats disponibles
    saved_results = list_results()

    if not saved_results:
        st.info("Aucun résultat de benchmark trouvé. Lancez un benchmark d'abord !")
        return

    # Sélection des résultats
    st.markdown("### 📂 Résultats disponibles")

    result_options = {
        r["filename"]: r for r in saved_results
    }

    selected_files = st.multiselect(
        "Sélectionnez un ou plusieurs résultats à afficher/comparer :",
        options=list(result_options.keys()),
        default=[list(result_options.keys())[0]],
    )

    if not selected_files:
        st.warning("Sélectionnez au moins un résultat.")
        return

    # Charger les résultats sélectionnés
    loaded_data = {}
    for fname in selected_files:
        filepath = result_options[fname]["filepath"]
        try:
            loaded_data[fname] = load_result(filepath)
        except Exception as e:
            st.error(f"Erreur chargement {fname}: {e}")

    if not loaded_data:
        return

    # ============================
    # AFFICHAGE SIMPLE (1 résultat)
    # ============================
    if len(loaded_data) == 1:
        fname, data = list(loaded_data.items())[0]
        _display_single_result(data, fname)

    # ============================
    # COMPARAISON (plusieurs résultats)
    # ============================
    else:
        _display_comparison(loaded_data)

    # Export
    st.markdown("---")
    st.markdown("### 📤 Export")
    for fname in selected_files:
        filepath = result_options[fname]["filepath"]
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"📥 Télécharger JSON - {fname}", key=f"dl_json_{fname}"):
                with open(filepath, "r") as f:
                    st.download_button(
                        label=f"💾 {fname}",
                        data=f.read(),
                        file_name=fname,
                        mime="application/json",
                        key=f"download_{fname}",
                    )
        with col2:
            if st.button(f"📥 Exporter CSV - {fname}", key=f"dl_csv_{fname}"):
                try:
                    csv_path = export_to_csv(filepath)
                    with open(csv_path, "r") as f:
                        st.download_button(
                            label=f"💾 {csv_path.name}",
                            data=f.read(),
                            file_name=csv_path.name,
                            mime="text/csv",
                            key=f"download_csv_{fname}",
                        )
                except Exception as e:
                    st.error(f"Erreur export CSV : {e}")


def _display_single_result(data: dict, filename: str):
    """Affiche les détails d'un seul résultat de benchmark."""
    import plotly.graph_objects as go
    import plotly.express as px

    st.markdown(f"### 🔍 Détails : {filename}")

    # Info machine
    hw = data.get("hardware", {})
    cpu_model = hw.get("cpu", {}).get("model", "Unknown")
    gpus = hw.get("gpu", {}).get("gpus", [])
    gpu_name = gpus[0]["name"] if gpus else "None"
    ram_total = hw.get("ram", {}).get("total_gb", 0)
    backend = hw.get("gpu", {}).get("primary_backend", "cpu")

    cols = st.columns(4)
    cols[0].metric("CPU", cpu_model)
    cols[1].metric("GPU", gpu_name)
    cols[2].metric("RAM", f"{ram_total} Go")
    cols[3].metric("Backend", backend.upper())

    st.markdown("---")

    # === Benchmarks Classiques ===
    classic = data.get("classic_benchmarks", {}).get("benchmarks", {})
    if classic:
        st.markdown("### ⚡ Benchmarks Classiques")

        # Graphique GFLOPS CPU
        cpu_data_chart = {"Test": [], "GFLOPS": [], "Type": []}

        cpu_st = classic.get("cpu_single_thread", {}).get("results", {})
        for size, vals in cpu_st.items():
            cpu_data_chart["Test"].append(size)
            cpu_data_chart["GFLOPS"].append(vals.get("gflops", 0))
            cpu_data_chart["Type"].append("Single-Thread")

        cpu_mt = classic.get("cpu_multi_thread", {}).get("results", {})
        for size, vals in cpu_mt.items():
            cpu_data_chart["Test"].append(size)
            cpu_data_chart["GFLOPS"].append(vals.get("gflops", 0))
            cpu_data_chart["Type"].append("Multi-Thread")

        if cpu_data_chart["Test"]:
            fig = px.bar(
                cpu_data_chart,
                x="Test", y="GFLOPS", color="Type",
                barmode="group",
                title="Performance CPU (GFLOPS) - Multiplication matricielle",
                color_discrete_map={"Single-Thread": "#FF6B6B", "Multi-Thread": "#4ECDC4"},
            )
            fig.update_layout(xaxis_title="Taille matrice", yaxis_title="GFLOPS")
            st.plotly_chart(fig, use_container_width=True)

        # Bandwidth mémoire
        mem = classic.get("memory_bandwidth", {}).get("results", {})
        if mem:
            mem_chart = {
                "Opération": ["Lecture", "Écriture", "Copie"],
                "Bande passante (Go/s)": [
                    mem.get("read", {}).get("bandwidth_gb_s", 0),
                    mem.get("write", {}).get("bandwidth_gb_s", 0),
                    mem.get("copy", {}).get("bandwidth_gb_s", 0),
                ],
            }
            fig = px.bar(
                mem_chart,
                x="Opération", y="Bande passante (Go/s)",
                title="Bande passante mémoire",
                color="Opération",
                color_discrete_sequence=["#667eea", "#764ba2", "#f093fb"],
            )
            st.plotly_chart(fig, use_container_width=True)

        # GPU
        gpu_bench = classic.get("gpu_compute", {})
        if gpu_bench.get("status") == "completed":
            gpu_results = gpu_bench.get("results", {})
            gpu_chart = {"Taille": [], "GFLOPS": []}
            for size, vals in gpu_results.items():
                gpu_chart["Taille"].append(size)
                gpu_chart["GFLOPS"].append(vals.get("gflops", 0))

            fig = px.bar(
                gpu_chart,
                x="Taille", y="GFLOPS",
                title=f"Performance GPU ({gpu_bench.get('backend', '')}) - {gpu_bench.get('device', '')}",
                color_discrete_sequence=["#FF6B6B"],
            )
            st.plotly_chart(fig, use_container_width=True)

        # Utilisation ressources
        resource = data.get("classic_benchmarks", {}).get("resource_usage", {})
        if resource and "cpu" in resource:
            st.markdown("**Utilisation des ressources pendant le benchmark :**")
            res_cols = st.columns(3)
            res_cols[0].metric("CPU moyen", f"{resource['cpu']['avg_percent']}%")
            res_cols[1].metric("CPU max", f"{resource['cpu']['max_percent']}%")
            res_cols[2].metric("RAM pic", f"{resource['ram'].get('peak_used_gb', 0)} Go")

    st.markdown("---")

    # === Benchmarks IA ===
    ai = data.get("ai_benchmarks", {})
    ai_results = ai.get("results", {})
    if ai_results:
        st.markdown("### 🤖 Benchmarks IA (Inférence LLM)")

        # Tableau récapitulatif
        table_data = {
            "Modèle": [], "Params": [], "Tokens/s": [],
            "1er token (s)": [], "Mémoire (Go)": [],
            "Stabilité": [], "Backend": [],
        }

        for key, model_data in ai_results.items():
            summary = model_data.get("summary", {})
            if summary:
                table_data["Modèle"].append(model_data.get("model", key))
                table_data["Params"].append(model_data.get("params", ""))
                table_data["Tokens/s"].append(summary.get("avg_tokens_per_second", 0))
                table_data["1er token (s)"].append(summary.get("avg_first_token_latency_s", 0))
                table_data["Mémoire (Go)"].append(summary.get("peak_memory_gb", 0))
                table_data["Stabilité"].append(summary.get("stability", "?"))
                backend_str = model_data.get("backend", {}).get("backend", "?")
                table_data["Backend"].append(backend_str.upper())
            elif model_data.get("status") == "skipped":
                table_data["Modèle"].append(model_data.get("model", key))
                table_data["Params"].append("")
                table_data["Tokens/s"].append(0)
                table_data["1er token (s)"].append(0)
                table_data["Mémoire (Go)"].append(0)
                table_data["Stabilité"].append("skipped")
                table_data["Backend"].append("")

        import pandas as pd
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Graphique Tokens/s
        active_models = {k: v for k, v in ai_results.items() if v.get("summary")}
        if active_models:
            tps_chart = {
                "Modèle": [],
                "Tokens/s": [],
            }
            for key, model_data in active_models.items():
                tps_chart["Modèle"].append(model_data.get("model", key))
                tps_chart["Tokens/s"].append(
                    model_data["summary"]["avg_tokens_per_second"]
                )

            fig = px.bar(
                tps_chart,
                x="Modèle", y="Tokens/s",
                title="Débit d'inférence par modèle",
                color="Modèle",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Graphique latence premier token
            ftl_chart = {
                "Modèle": [],
                "Latence (s)": [],
            }
            for key, model_data in active_models.items():
                ftl_chart["Modèle"].append(model_data.get("model", key))
                ftl_chart["Latence (s)"].append(
                    model_data["summary"]["avg_first_token_latency_s"]
                )

            fig = px.bar(
                ftl_chart,
                x="Modèle", y="Latence (s)",
                title="Latence du premier token par modèle",
                color="Modèle",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # JSON brut
    with st.expander("📋 Données brutes (JSON)"):
        st.json(data)


def _display_comparison(loaded_data: dict):
    """Affiche la comparaison entre plusieurs résultats avec couleurs distinctes."""
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd

    st.markdown("### ⚖️ Comparaison des résultats")

    # ─── Palette de couleurs distinctes et stables ───
    COLORS = [
        "#636EFA",  # bleu
        "#EF553B",  # rouge
        "#00CC96",  # vert
        "#AB63FA",  # violet
        "#FFA15A",  # orange
        "#19D3F3",  # cyan
        "#FF6692",  # rose
        "#B6E880",  # vert clair
    ]

    # ─── Construire des labels uniques par résultat ───
    result_labels = {}   # fname → label court unique
    result_colors = {}   # fname → couleur
    for idx, (fname, data) in enumerate(loaded_data.items()):
        hw = data.get("hardware", {})
        cpu_short = hw.get("cpu", {}).get("model", "Unknown")
        # Extraire un nom court du CPU (ex: "Apple M1", "i9-13900K")
        for prefix in ["Apple ", "Intel(R) Core(TM) ", "AMD Ryzen "]:
            if prefix in cpu_short:
                cpu_short = cpu_short.split(prefix)[-1].split(" @")[0].split(" CPU")[0]
                break
        timestamp = data.get("timestamp", "")[:16].replace("T", " ")
        label = f"{cpu_short} ({timestamp})"
        # Éviter les doublons exacts
        if label in result_labels.values():
            label += f" #{idx+1}"
        result_labels[fname] = label
        result_colors[fname] = COLORS[idx % len(COLORS)]

    # ─── Afficher la légende des couleurs ───
    legend_html = " &nbsp; ".join(
        f'<span style="display:inline-block;width:14px;height:14px;'
        f'background:{color};border-radius:3px;margin-right:4px;'
        f'vertical-align:middle;"></span>'
        f'<span style="vertical-align:middle;font-weight:600;">{label}</span>'
        for fname, label, color in [
            (f, result_labels[f], result_colors[f]) for f in loaded_data
        ]
    )
    st.markdown(
        f'<div style="background:#f0f2f6;padding:10px 16px;border-radius:8px;'
        f'margin-bottom:20px;">'
        f'<b>🎨 Légende :</b> &nbsp; {legend_html}</div>',
        unsafe_allow_html=True,
    )

    # ─── Tableau comparatif matériel ───
    st.markdown("#### 🖥️ Comparaison matérielle")
    hw_table = {"Résultat": [], "CPU": [], "GPU": [], "RAM (Go)": [], "Backend": []}
    for fname, data in loaded_data.items():
        hw = data.get("hardware", {})
        hw_table["Résultat"].append(result_labels[fname])
        hw_table["CPU"].append(hw.get("cpu", {}).get("model", "?"))
        gpus = hw.get("gpu", {}).get("gpus", [])
        hw_table["GPU"].append(gpus[0]["name"] if gpus else "None")
        hw_table["RAM (Go)"].append(hw.get("ram", {}).get("total_gb", 0))
        hw_table["Backend"].append(
            hw.get("gpu", {}).get("primary_backend", "cpu").upper()
        )
    st.dataframe(pd.DataFrame(hw_table), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════
    # Comparaison CPU GFLOPS
    # ══════════════════════════════════════════════
    st.markdown("#### ⚡ Comparaison CPU")

    # Collecter CPU ST et MT pour chaque résultat
    has_cpu = False
    fig_cpu = go.Figure()
    for fname, data in loaded_data.items():
        classic = data.get("classic_benchmarks", {}).get("benchmarks", {})
        cpu_st = classic.get("cpu_single_thread", {}).get("results", {})
        cpu_mt = classic.get("cpu_multi_thread", {}).get("results", {})

        if cpu_st:
            largest_key = list(cpu_st.keys())[-1]
            st_gflops = cpu_st[largest_key].get("gflops", 0)
        else:
            st_gflops = 0

        if cpu_mt:
            largest_key = list(cpu_mt.keys())[-1]
            mt_gflops = cpu_mt[largest_key].get("gflops", 0)
        else:
            mt_gflops = 0

        if st_gflops or mt_gflops:
            has_cpu = True
            fig_cpu.add_trace(go.Bar(
                name=result_labels[fname],
                x=["Single-Thread", "Multi-Thread"],
                y=[st_gflops, mt_gflops],
                marker_color=result_colors[fname],
                text=[f"{st_gflops:.1f}", f"{mt_gflops:.1f}"],
                textposition="outside",
            ))

    if has_cpu:
        fig_cpu.update_layout(
            barmode="group",
            title="Performance CPU — GFLOPS (plus grande matrice)",
            yaxis_title="GFLOPS",
            legend_title="Résultat",
        )
        st.plotly_chart(fig_cpu, use_container_width=True)

    # ══════════════════════════════════════════════
    # Comparaison Mémoire
    # ══════════════════════════════════════════════
    has_mem = False
    fig_mem = go.Figure()
    for fname, data in loaded_data.items():
        mem = data.get("classic_benchmarks", {}).get("benchmarks", {}).get(
            "memory_bandwidth", {}
        ).get("results", {})
        if mem:
            has_mem = True
            fig_mem.add_trace(go.Bar(
                name=result_labels[fname],
                x=["Lecture", "Écriture", "Copie"],
                y=[
                    mem.get("read", {}).get("bandwidth_gb_s", 0),
                    mem.get("write", {}).get("bandwidth_gb_s", 0),
                    mem.get("copy", {}).get("bandwidth_gb_s", 0),
                ],
                marker_color=result_colors[fname],
                text=[
                    f"{mem.get('read', {}).get('bandwidth_gb_s', 0):.1f}",
                    f"{mem.get('write', {}).get('bandwidth_gb_s', 0):.1f}",
                    f"{mem.get('copy', {}).get('bandwidth_gb_s', 0):.1f}",
                ],
                textposition="outside",
            ))

    if has_mem:
        st.markdown("#### 🧠 Comparaison Mémoire")
        fig_mem.update_layout(
            barmode="group",
            title="Bande passante mémoire (Go/s)",
            yaxis_title="Go/s",
            legend_title="Résultat",
        )
        st.plotly_chart(fig_mem, use_container_width=True)

    # ══════════════════════════════════════════════
    # Comparaison GPU
    # ══════════════════════════════════════════════
    has_gpu = False
    fig_gpu = go.Figure()
    for fname, data in loaded_data.items():
        gpu_bench = data.get("classic_benchmarks", {}).get("benchmarks", {}).get(
            "gpu_compute", {}
        )
        if gpu_bench.get("status") == "completed":
            gpu_results = gpu_bench.get("results", {})
            if gpu_results:
                has_gpu = True
                sizes = list(gpu_results.keys())
                gflops_vals = [gpu_results[s].get("gflops", 0) for s in sizes]
                fig_gpu.add_trace(go.Bar(
                    name=result_labels[fname],
                    x=sizes,
                    y=gflops_vals,
                    marker_color=result_colors[fname],
                    text=[f"{v:.0f}" for v in gflops_vals],
                    textposition="outside",
                ))

    if has_gpu:
        st.markdown("#### 🎮 Comparaison GPU")
        fig_gpu.update_layout(
            barmode="group",
            title="Performance GPU (GFLOPS)",
            yaxis_title="GFLOPS",
            legend_title="Résultat",
        )
        st.plotly_chart(fig_gpu, use_container_width=True)

    # ══════════════════════════════════════════════
    # Comparaison Inférence IA
    # ══════════════════════════════════════════════
    st.markdown("#### 🤖 Comparaison Inférence IA")

    # Collecter tous les modèles testés
    all_models = {}
    for fname, data in loaded_data.items():
        ai_results = data.get("ai_benchmarks", {}).get("results", {})
        for model_key, model_data in ai_results.items():
            if model_data.get("summary"):
                model_name = model_data.get("model", model_key)
                all_models[model_key] = model_name

    if all_models:
        # ── Graphique global : Tokens/s par modèle ──
        fig_tps = go.Figure()
        fig_ftl = go.Figure()
        model_keys_sorted = sorted(all_models.keys())
        model_names_sorted = [all_models[k] for k in model_keys_sorted]

        for fname, data in loaded_data.items():
            ai_results = data.get("ai_benchmarks", {}).get("results", {})
            tps_values = []
            ftl_values = []
            for model_key in model_keys_sorted:
                model_data = ai_results.get(model_key, {})
                summary = model_data.get("summary", {})
                tps_values.append(summary.get("avg_tokens_per_second", 0))
                ftl_values.append(summary.get("avg_first_token_latency_s", 0))

            fig_tps.add_trace(go.Bar(
                name=result_labels[fname],
                x=model_names_sorted,
                y=tps_values,
                marker_color=result_colors[fname],
                text=[f"{v:.1f}" if v > 0 else "" for v in tps_values],
                textposition="outside",
            ))
            fig_ftl.add_trace(go.Bar(
                name=result_labels[fname],
                x=model_names_sorted,
                y=ftl_values,
                marker_color=result_colors[fname],
                text=[f"{v:.3f}" if v > 0 else "" for v in ftl_values],
                textposition="outside",
            ))

        col1, col2 = st.columns(2)
        with col1:
            fig_tps.update_layout(
                barmode="group",
                title="Débit d'inférence (tokens/s)",
                yaxis_title="Tokens/s",
                legend_title="Résultat",
                xaxis_title="Modèle",
            )
            st.plotly_chart(fig_tps, use_container_width=True)

        with col2:
            fig_ftl.update_layout(
                barmode="group",
                title="Latence du premier token (s)",
                yaxis_title="Secondes",
                legend_title="Résultat",
                xaxis_title="Modèle",
            )
            st.plotly_chart(fig_ftl, use_container_width=True)

        # ── Graphique mémoire pic par modèle ──
        fig_mem_ai = go.Figure()
        for fname, data in loaded_data.items():
            ai_results = data.get("ai_benchmarks", {}).get("results", {})
            mem_values = []
            for model_key in model_keys_sorted:
                model_data = ai_results.get(model_key, {})
                summary = model_data.get("summary", {})
                mem_values.append(summary.get("peak_memory_gb", 0))

            fig_mem_ai.add_trace(go.Bar(
                name=result_labels[fname],
                x=model_names_sorted,
                y=mem_values,
                marker_color=result_colors[fname],
                text=[f"{v:.2f}" if v > 0 else "" for v in mem_values],
                textposition="outside",
            ))

        fig_mem_ai.update_layout(
            barmode="group",
            title="Mémoire pic par modèle (Go)",
            yaxis_title="Go",
            legend_title="Résultat",
            xaxis_title="Modèle",
        )
        st.plotly_chart(fig_mem_ai, use_container_width=True)

        # ── Tableau comparatif IA complet ──
        st.markdown("#### 📋 Tableau comparatif complet")
        ia_table = {
            "Résultat": [], "Modèle": [], "Tokens/s": [],
            "Latence 1er token (s)": [], "Mémoire pic (Go)": [],
            "Stabilité": [],
        }
        for fname, data in loaded_data.items():
            ai_results = data.get("ai_benchmarks", {}).get("results", {})
            for model_key, model_data in ai_results.items():
                summary = model_data.get("summary", {})
                if summary:
                    ia_table["Résultat"].append(result_labels[fname])
                    ia_table["Modèle"].append(model_data.get("model", model_key))
                    ia_table["Tokens/s"].append(summary.get("avg_tokens_per_second", 0))
                    ia_table["Latence 1er token (s)"].append(
                        summary.get("avg_first_token_latency_s", 0)
                    )
                    ia_table["Mémoire pic (Go)"].append(
                        summary.get("peak_memory_gb", 0)
                    )
                    ia_table["Stabilité"].append(summary.get("stability", "?"))

        if ia_table["Résultat"]:
            st.dataframe(
                pd.DataFrame(ia_table), use_container_width=True, hide_index=True
            )
    else:
        st.info("Aucun résultat d'inférence IA à comparer.")


# =============================================================================
# Router
# =============================================================================
if page == "🏠 Matériel":
    page_hardware()
elif page == "🚀 Benchmark":
    page_benchmark()
elif page == "📊 Résultats":
    page_results()
