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
    get_available_quantizations,
    get_compatible_quantizations,
    is_quantization_downloaded,
    download_quantization,
    delete_quantization,
)
from src.llama_server import (
    find_llama_server_binary,
    LlamaServerManager,
    check_server_status,
    run_all_server_benchmarks,
    get_llama_cpp_releases_url,
)
from src.results_manager import (
    save_results,
    list_results,
    load_result,
    compare_results,
    export_to_csv,
)
from src.config import AVAILABLE_MODELS, QUANTIZATION_VARIANTS, RESULTS_DIR

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
    /* Boutons supprimer compacts */
    [data-testid="stButton"] button[kind="secondary"] {
        min-height: 0;
    }
    [data-testid="column"]:last-child [data-testid="stButton"] button {
        padding: 0.15rem 0.5rem;
        font-size: 0.75rem;
        line-height: 1;
        min-height: 0;
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
if "quant_results" not in st.session_state:
    st.session_state.quant_results = None
if "server_mode" not in st.session_state:
    st.session_state.server_mode = False
if "server_binary_path" not in st.session_state:
    st.session_state.server_binary_path = ""
if "server_host" not in st.session_state:
    st.session_state.server_host = "127.0.0.1"
if "server_port" not in st.session_state:
    st.session_state.server_port = 8080
if "server_auto_mode" not in st.session_state:
    st.session_state.server_auto_mode = True


# =============================================================================
# Sidebar Navigation
# =============================================================================
st.sidebar.markdown("## ComputeLLM")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Matériel", "Benchmark", "Résultats"],
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
    st.markdown('<h1 class="main-header">Détection Matérielle</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyse automatique de votre configuration</p>', unsafe_allow_html=True)

    # Bouton de détection
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Détecter le matériel", use_container_width=True, type="primary"):
            with st.spinner("Analyse du matériel en cours..."):
                st.session_state.hardware_info = get_full_hardware_info()

    if st.session_state.hardware_info is None:
        st.info("Cliquez sur le bouton ci-dessus pour détecter votre matériel.")
        return

    hw = st.session_state.hardware_info

    # --- Système d'exploitation ---
    st.markdown("### Système d'exploitation")
    os_info = hw["os"]
    cols = st.columns(4)
    cols[0].metric("OS", os_info["system"])
    cols[1].metric("Version", os_info["release"])
    cols[2].metric("Architecture", os_info["architecture"])
    cols[3].metric("Python", os_info["python_version"])

    st.markdown("---")

    # --- CPU ---
    st.markdown("### Processeur (CPU)")
    cpu = hw["cpu"]
    cols = st.columns(4)
    cols[0].metric("Modèle", cpu.get("model", "Unknown"))
    cols[1].metric("Cœurs physiques", cpu.get("physical_cores", "?"))
    cols[2].metric("Cœurs logiques", cpu.get("logical_cores", "?"))
    arch_type = cpu.get("architecture_type", cpu.get("architecture", "?"))
    cols[3].metric("Architecture", arch_type)

    if cpu.get("is_apple_silicon"):
        cols2 = st.columns(4)
        cols2[0].metric("Type", "Apple Silicon")
        perf = cpu.get("performance_cores", "?")
        eff = cpu.get("efficiency_cores", "?")
        cols2[1].metric("Cœurs Performance", perf)
        cols2[2].metric("Cœurs Efficience", eff)
        if cpu.get("frequency_mhz"):
            cols2[3].metric("Fréquence", f"{cpu['frequency_mhz'].get('current', '?')} MHz")

    st.markdown("---")

    # --- Mémoire RAM ---
    st.markdown("### Mémoire RAM")
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
    st.markdown("### GPU & Accélération")
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
        py_cols = st.columns(4)
        if py_backends.get("llama_cpp"):
            py_cols[0].success(f"llama-cpp-python {py_backends.get('llama_cpp_version', '')}")
        else:
            py_cols[0].error("llama-cpp-python non installé")

        if py_backends.get("pytorch"):
            ver = py_backends.get("pytorch_version", "")
            cuda_str = f" (CUDA {py_backends['pytorch_cuda_version']})" if py_backends.get("pytorch_cuda") else ""
            mps_str = " (MPS)" if py_backends.get("pytorch_mps") else ""
            xpu_str = " (XPU)" if py_backends.get("pytorch_xpu") else ""
            py_cols[1].success(f"PyTorch {ver}{cuda_str}{mps_str}{xpu_str}")
        else:
            py_cols[1].warning("PyTorch non installé (GPU benchmark indisponible)")

        if py_backends.get("llama_server"):
            py_cols[2].success(f"llama-server détecté")
            if py_backends.get("llama_server_path"):
                st.caption(f"Chemin : {py_backends['llama_server_path']}")
        else:
            py_cols[2].info("llama-server non trouvé (optionnel)")

        if py_backends.get("ipex"):
            py_cols[3].success(f"IPEX {py_backends.get('ipex_version', '')}")
        elif py_backends.get("pytorch_xpu"):
            py_cols[3].info("IPEX non détecté (XPU via PyTorch)")

    # Modèles compatibles
    st.markdown("---")
    st.markdown("### Modèles LLM compatibles")
    ram_total = ram["total_gb"]
    compatible = get_compatible_models(ram_total)

    for key, model in AVAILABLE_MODELS.items():
        is_compat = key in compatible
        is_downloaded = is_model_downloaded(key)
        icon = "" if is_compat else ""
        dl_icon = "" if is_downloaded else ""

        cols = st.columns([0.5, 2, 1, 1, 1, 1])
        cols[0].write(icon)
        cols[1].write(f"**{model['name']}** ({model['params']})")
        cols[2].write(f"{model['size_gb']} Go")
        cols[3].write(f"RAM min: {model['min_ram_gb']} Go")
        cols[4].write(dl_icon + (" Téléchargé" if is_downloaded else " Non téléchargé"))
        cols[5].write("Compatible" if is_compat else "Incompatible")

        # Détail des quantifications
        variants = get_available_quantizations(key)
        if variants:
            with st.expander(f"Quantifications disponibles — {model['name']}", expanded=False):
                for qk, qv in variants.items():
                    q_downloaded = is_quantization_downloaded(key, qk)
                    q_compat = qv["min_ram_gb"] <= ram_total

                    q_cols = st.columns([0.3, 1.2, 0.8, 0.8, 0.8, 0.6])
                    q_cols[0].write("" if q_compat else "")
                    q_cols[1].write(f"**{qk}** ({qv['bits']}-bit)")
                    q_cols[2].write(f"{qv['size_gb']} Go")
                    q_cols[3].write(f"RAM min: {qv['min_ram_gb']} Go")

                    if q_downloaded:
                        q_cols[4].write("Installé")
                        if q_cols[5].button("✕", key=f"del_{key}_{qk}", help=f"Supprimer {model['name']} {qk}"):
                            delete_quantization(key, qk)
                            st.rerun()
                    else:
                        q_cols[4].write("Non installé")
                        q_cols[5].write("")

    # Export JSON brut
    with st.expander("Données brutes (JSON)"):
        st.json(hw)


# =============================================================================
# PAGE 2 : Benchmark
# =============================================================================
def page_benchmark():
    st.markdown('<h1 class="main-header">Benchmark</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Lancer l\'ensemble des benchmarks matériels et IA</p>', unsafe_allow_html=True)

    # Vérifier que le matériel a été détecté
    if st.session_state.hardware_info is None:
        st.warning("Veuillez d'abord détecter le matériel dans la page 'Matériel'.")
        if st.button("Détecter le matériel maintenant"):
            with st.spinner("Détection..."):
                st.session_state.hardware_info = get_full_hardware_info()
            st.rerun()
        return

    hw = st.session_state.hardware_info

    # Configuration des benchmarks
    st.markdown("### Configuration")

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

    # ─── Comparaison de quantification ───
    st.markdown("---")
    st.markdown("### Comparaison de quantification")
    st.caption(
        "Comparez les performances de différentes quantifications (Q2, Q3, Q4, Q5, Q6, Q8) "
        "d'un même modèle. Mesure l'impact sur les tokens/s, la latence, la mémoire et "
        "le temps de chargement."
    )

    quant_selections = {}  # model_key → [quant_key, ...]
    models_with_quants = {
        k: v for k, v in QUANTIZATION_VARIANTS.items()
        if k in compatible
    }

    if models_with_quants:
        for model_key, variants in models_with_quants.items():
            model_name = AVAILABLE_MODELS[model_key]["name"]
            params = AVAILABLE_MODELS[model_key]["params"]

            with st.expander(f"🔧 {model_name} ({params})", expanded=(model_key == "tinyllama-1.1b")):
                enable_quant = st.checkbox(
                    f"Activer la comparaison de quantification pour {model_name}",
                    value=False,
                    key=f"quant_enable_{model_key}",
                )

                if enable_quant:
                    compatible_quants = get_compatible_quantizations(model_key, ram_total)
                    selected_quants = []

                    q_cols = st.columns(min(len(compatible_quants), 6))
                    for i, (qk, qv) in enumerate(compatible_quants.items()):
                        col_idx = i % len(q_cols)
                        downloaded = is_quantization_downloaded(model_key, qk)
                        dl_str = "✅" if downloaded else "📥"
                        with q_cols[col_idx]:
                            if st.checkbox(
                                f"{dl_str} {qk} ({qv['size_gb']} Go)",
                                value=downloaded,
                                key=f"quant_{model_key}_{qk}",
                                help=qv["description"],
                            ):
                                selected_quants.append(qk)

                    if selected_quants:
                        quant_selections[model_key] = selected_quants
                        st.info(f"{len(selected_quants)} quantification(s) sélectionnée(s) : {', '.join(selected_quants)}")

    else:
        st.info("Aucun modèle compatible pour la comparaison de quantification.")

    # ─── Mode d'inférence : llama-cpp-python vs llama-server ───
    st.markdown("---")
    st.markdown("### Mode d'inférence")
    st.caption(
        "**llama-cpp-python** : bindings Python (nécessite compilation avec le bon backend). "
        "**llama-server** : binaire pré-compilé (aucune compilation côté Python, "
        "[téléchargement ici]({releases_url})).".format(releases_url=get_llama_cpp_releases_url())
    )

    server_mode = st.toggle(
        "Utiliser llama-server (mode HTTP)",
        value=st.session_state.server_mode,
        key="server_mode_toggle",
        help="Utilise un binaire llama-server pré-compilé au lieu de llama-cpp-python",
    )
    st.session_state.server_mode = server_mode

    if server_mode:
        srv_col1, srv_col2 = st.columns(2)

        with srv_col1:
            auto_mode = st.radio(
                "Mode serveur",
                ["Auto (ComputeLLM démarre le serveur)", "Manuel (serveur externe)"],
                index=0 if st.session_state.server_auto_mode else 1,
                key="server_mode_radio",
            )
            st.session_state.server_auto_mode = auto_mode.startswith("Auto")

        with srv_col2:
            if st.session_state.server_auto_mode:
                # Détection automatique du binaire
                detected_binary = find_llama_server_binary()
                default_path = detected_binary or st.session_state.server_binary_path

                binary_path = st.text_input(
                    "Chemin vers llama-server",
                    value=default_path,
                    placeholder="C:/chemin/vers/llama-server.exe ou /chemin/vers/llama-server",
                    key="server_binary_input",
                    help="Chemin vers le binaire llama-server (fichier .exe ou dossier contenant le binaire).",
                )

                # Nettoyer le chemin (espaces, guillemets)
                if binary_path:
                    binary_path = binary_path.strip().strip('"').strip("'").strip()

                # Résoudre le chemin : fichier direct ou dossier contenant le binaire
                if binary_path:
                    resolved = find_llama_server_binary(custom_path=binary_path)
                    if resolved:
                        binary_path = resolved
                        st.success(f"Binaire trouvé : `{binary_path}`")
                    else:
                        # Le chemin brut sera testé au lancement
                        import os
                        if os.path.isfile(binary_path):
                            st.success(f"Binaire : `{binary_path}`")
                        elif os.path.isdir(binary_path):
                            st.error(
                                f"Dossier trouvé mais aucun llama-server(.exe) dedans.\n"
                                f"Vérifiez que le binaire est bien dans : `{binary_path}`"
                            )
                        else:
                            st.error(
                                f"Fichier introuvable : `{binary_path}`\n"
                                "Vérifiez le chemin (copier-coller depuis l'explorateur de fichiers)."
                            )

                st.session_state.server_binary_path = binary_path

                if not binary_path:
                    st.warning(
                        "Binaire llama-server introuvable. "
                        f"[Télécharger les binaires pré-compilés]({get_llama_cpp_releases_url()})"
                    )
            else:
                # Mode manuel — connexion à un serveur existant
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    host = st.text_input(
                        "Host", value=st.session_state.server_host, key="server_host_input"
                    )
                    st.session_state.server_host = host
                with m_col2:
                    port = st.number_input(
                        "Port", value=st.session_state.server_port,
                        min_value=1, max_value=65535, key="server_port_input"
                    )
                    st.session_state.server_port = int(port)

                # Vérifier le statut du serveur
                if st.button("Vérifier la connexion", key="check_server"):
                    status = check_server_status(host, int(port))
                    if status["running"]:
                        st.success(f"Serveur actif sur {status['url']}")
                        if status.get("info"):
                            st.json(status["info"])
                    else:
                        st.error(f"Aucun serveur trouvé sur {status['url']}")

    st.markdown("---")

    # Résumé de la configuration
    backend_info = detect_best_backend()
    st.markdown("### Résumé")

    n_quant_tests = sum(len(qs) for qs in quant_selections.values())
    n_tests = sum([run_classic, run_classic_mt, run_memory, run_gpu]) + len(selected_models) + n_quant_tests

    inference_mode_label = "llama-server" if server_mode else "llama-cpp-python"
    cols = st.columns(6)
    cols[0].metric("Backend IA", backend_info["backend"].upper())
    cols[1].metric("Mode inférence", inference_mode_label)
    cols[2].metric("Modèles sélectionnés", len(selected_models))
    cols[3].metric("Tests quantification", n_quant_tests)
    cols[4].metric("RAM disponible", f"{hw['ram']['available_gb']} Go")
    cols[5].metric("Tests total", n_tests)

    st.markdown("---")

    # ===== BOUTON UNIQUE DE LANCEMENT =====
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        launch = st.button(
            "LANCER TOUS LES BENCHMARKS",
            use_container_width=True,
            type="primary",
            disabled=st.session_state.benchmark_running,
        )

    if launch:
        st.session_state.benchmark_running = True
        st.session_state.classic_results = None
        st.session_state.ai_results = None
        st.session_state.quant_results = None

        total_start = time.time()

        # ============================
        # BENCHMARKS CLASSIQUES
        # ============================
        if any([run_classic, run_classic_mt, run_memory, run_gpu]):
            st.markdown("### Benchmarks Classiques")
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
                        label=f"Benchmarks classiques terminés ({classic_results['total_time_s']:.1f}s)",
                        state="complete"
                    )
                except Exception as e:
                    classic_status.update(label=f"Erreur : {e}", state="error")
                    st.error(f"Erreur benchmarks classiques : {e}")

            classic_progress.progress(1.0)
        else:
            st.info("Benchmarks classiques désactivés.")

        # ============================
        # BENCHMARKS IA (modèles + quantification)
        # ============================
        has_ai_work = bool(selected_models) or bool(quant_selections)

        if has_ai_work:
            mode_label = "serveur" if server_mode else "llama-cpp-python"
            st.markdown(f"### Benchmarks IA (Inférence LLM — {mode_label})")
            ai_progress = st.progress(0.0)
            ai_status = st.status("Benchmarks IA en cours...", expanded=True)

            def ai_callback(p, msg):
                ai_progress.progress(min(p, 1.0))
                ai_status.update(label=msg)

            with ai_status:
                # Téléchargement des modèles de base si nécessaire
                for model_key in selected_models:
                    if not is_model_downloaded(model_key):
                        model_info = AVAILABLE_MODELS[model_key]
                        st.write(f"Téléchargement de {model_info['name']}...")
                        try:
                            download_model(model_key)
                            st.write(f"✅ {model_info['name']} téléchargé.")
                        except Exception as e:
                            st.error(f"Erreur téléchargement {model_info['name']}: {e}")

                # Téléchargement des quantifications si nécessaire
                for model_key, quant_keys in quant_selections.items():
                    model_info = AVAILABLE_MODELS[model_key]
                    for qk in quant_keys:
                        if not is_quantization_downloaded(model_key, qk):
                            st.write(f"Téléchargement de {model_info['name']} {qk}...")
                            try:
                                download_quantization(model_key, qk)
                                st.write(f"✅ {model_info['name']} {qk} téléchargé.")
                            except Exception as e:
                                st.error(f"Erreur téléchargement {model_info['name']} {qk}: {e}")

                # Exécution des benchmarks
                st.write("Exécution des inférences...")
                try:
                    if server_mode:
                        # ── Mode llama-server ──
                        if st.session_state.server_auto_mode:
                            srv_binary = st.session_state.server_binary_path
                            if not srv_binary:
                                raise RuntimeError(
                                    "Aucun binaire llama-server configuré. "
                                    "Configurez le chemin ou téléchargez-le depuis "
                                    f"{get_llama_cpp_releases_url()}"
                                )
                            srv = LlamaServerManager(
                                binary_path=srv_binary,
                                host=st.session_state.server_host,
                                port=st.session_state.server_port,
                            )
                        else:
                            srv = LlamaServerManager(
                                binary_path=None,
                                host=st.session_state.server_host,
                                port=st.session_state.server_port,
                            )

                        ai_results = run_all_server_benchmarks(
                            model_keys=selected_models if selected_models else [],
                            quantization_models=quant_selections if quant_selections else None,
                            server_manager=srv,
                            progress_callback=ai_callback,
                        )
                    else:
                        # ── Mode llama-cpp-python classique ──
                        ai_results = run_all_ai_benchmarks(
                            model_keys=selected_models if selected_models else [],
                            quantization_models=quant_selections if quant_selections else None,
                            progress_callback=ai_callback,
                        )

                    st.session_state.ai_results = ai_results

                    # Vérifier les erreurs individuelles par modèle
                    model_errors = []
                    for mk, mr in ai_results.get("results", {}).items():
                        if mr.get("status") == "error":
                            model_errors.append((mr.get("model", mk), mr.get("error", "Erreur inconnue")))
                    for mk, comp in ai_results.get("quantization_comparison", {}).items():
                        for qk, qr in comp.get("results", {}).items():
                            if qr.get("status") == "error":
                                model_errors.append((f"{comp.get('model_name', mk)} {qk}", qr.get("error", "Erreur inconnue")))

                    if model_errors:
                        for m_name, m_err in model_errors:
                            st.error(f"❌ **{m_name}** : {m_err}")
                        ai_status.update(
                            label=f"Benchmarks IA terminés avec {len(model_errors)} erreur(s) ({ai_results['total_time_s']:.1f}s)",
                            state="error" if len(model_errors) == len(ai_results.get('results', {})) else "complete"
                        )
                    else:
                        ai_status.update(
                            label=f"Benchmarks IA terminés ({ai_results['total_time_s']:.1f}s)",
                            state="complete"
                        )
                except Exception as e:
                    ai_status.update(label=f"Erreur : {e}", state="error")
                    st.error(f"Erreur benchmarks IA : {e}")

            ai_progress.progress(1.0)
        else:
            st.info("Aucun modèle IA ni comparaison de quantification sélectionné.")

        # ============================
        # SAUVEGARDE AUTOMATIQUE
        # ============================
        total_time = time.time() - total_start

        st.markdown("---")
        st.markdown("### Sauvegarde")

        try:
            save_path = save_results(
                hardware_info=hw,
                classic_results=st.session_state.classic_results,
                ai_results=st.session_state.ai_results,
            )
            st.session_state.last_save_path = str(save_path)
            st.success(f"Résultats sauvegardés : `{save_path.name}`")
        except Exception as e:
            st.error(f"Erreur sauvegarde : {e}")

        st.markdown(f"**Temps total : {total_time:.1f} secondes**")

        st.session_state.benchmark_running = False
        st.balloons()

    # Afficher un résumé rapide si des résultats sont en session
    if st.session_state.classic_results or st.session_state.ai_results:
        st.markdown("---")
        st.markdown("### Derniers résultats")

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

            # Résumé rapide de la comparaison de quantification
            quant_comp = st.session_state.ai_results.get("quantization_comparison", {})
            if quant_comp:
                st.markdown("#### Aperçu comparaison quantification")
                for model_key, comp_data in quant_comp.items():
                    model_name = comp_data.get("model_name", model_key)
                    table = comp_data.get("comparison_table", [])
                    if table:
                        n_q = len(table)
                        cols = st.columns(n_q)
                        for i, row in enumerate(table):
                            tps = row.get("tokens_per_second", 0)
                            cols[i].metric(
                                f"{model_name} {row['quantization']}",
                                f"{tps} tok/s",
                                f"{row.get('file_size_gb', 0):.2f} Go",
                            )

        st.info("Consultez la page **Résultats** pour une analyse détaillée.")


# =============================================================================
# PAGE 3 : Résultats
# =============================================================================
def page_results():
    st.markdown('<h1 class="main-header">Résultats</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Visualisation et comparaison des benchmarks</p>', unsafe_allow_html=True)

    # Charger la liste des résultats disponibles
    saved_results = list_results()

    if not saved_results:
        st.info("Aucun résultat de benchmark trouvé. Lancez un benchmark d'abord !")
        return

    # Sélection des résultats
    st.markdown("### Résultats disponibles")

    result_options = {
        r["filename"]: r for r in saved_results
    }

    # Afficher chaque résultat avec une checkbox et un bouton supprimer
    selected_files = []
    for fname, meta in result_options.items():
        cpu = meta.get("cpu", "Unknown")
        timestamp = meta.get("timestamp", "")[:16].replace("T", " ")
        ram = meta.get("ram_gb", 0)
        backend = meta.get("backend", "cpu").upper()
        label = f"**{fname}** — {cpu} | {ram} Go RAM | {backend} | {timestamp}"

        col_cb, col_del = st.columns([20, 1])
        with col_cb:
            if st.checkbox(label, value=False, key=f"cb_{fname}"):
                selected_files.append(fname)
        with col_del:
            if st.button("✕", key=f"del_result_{fname}", help=f"Supprimer {fname}"):
                try:
                    os.remove(meta["filepath"])
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur suppression : {e}")

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
    st.markdown("### Export")
    for fname in selected_files:
        filepath = result_options[fname]["filepath"]
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"Télécharger JSON - {fname}", key=f"dl_json_{fname}"):
                with open(filepath, "r") as f:
                    st.download_button(
                        label=f"💾 {fname}",
                        data=f.read(),
                        file_name=fname,
                        mime="application/json",
                        key=f"download_{fname}",
                    )
        with col2:
            if st.button(f"Exporter CSV - {fname}", key=f"dl_csv_{fname}"):
                try:
                    csv_path = export_to_csv(filepath)
                    with open(csv_path, "r") as f:
                        st.download_button(
                            label=f"{csv_path.name}",
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

    st.markdown(f"### Détails : {filename}")

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
        st.markdown("### Benchmarks Classiques")

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
        st.markdown("### Benchmarks IA (Inférence LLM)")

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
            elif model_data.get("status") == "error":
                table_data["Modèle"].append(model_data.get("model", key))
                table_data["Params"].append(model_data.get("params", ""))
                table_data["Tokens/s"].append(0)
                table_data["1er token (s)"].append(0)
                table_data["Mémoire (Go)"].append(0)
                table_data["Stabilité"].append("❌ erreur")
                table_data["Backend"].append("")

        import pandas as pd
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Afficher les erreurs détaillées
        error_models = {k: v for k, v in ai_results.items() if v.get("status") == "error"}
        if error_models:
            st.markdown("**⚠️ Modèles en erreur :**")
            for key, model_data in error_models.items():
                err_msg = model_data.get("error", "Erreur inconnue")
                st.error(f"**{model_data.get('model', key)}** : {err_msg}")

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

    # === Comparaison de Quantification ===
    quant_comp = ai.get("quantization_comparison", {})
    if quant_comp:
        st.markdown("---")
        st.markdown("### Comparaison de Quantification")

        import plotly.graph_objects as go

        for model_key, comp_data in quant_comp.items():
            model_name = comp_data.get("model_name", model_key)
            params = comp_data.get("params", "")
            table = comp_data.get("comparison_table", [])
            qresults = comp_data.get("results", {})

            if not table:
                st.warning(f"Aucun résultat de quantification pour {model_name}.")
                continue

            st.markdown(f"#### {model_name} ({params})")

            # Tableau comparatif
            qt_data = {
                "Quantification": [],
                "Bits": [],
                "Taille fichier (Go)": [],
                "Tokens/s": [],
                "1er token (s)": [],
                "Latence inter-token (ms)": [],
                "Mémoire pic (Go)": [],
                "Chargement (s)": [],
                "Stabilité": [],
            }
            for row in table:
                qt_data["Quantification"].append(row["quantization"])
                qt_data["Bits"].append(row["bits"])
                qt_data["Taille fichier (Go)"].append(round(row.get("file_size_gb") or 0, 3))
                qt_data["Tokens/s"].append(row.get("tokens_per_second") or 0)
                qt_data["1er token (s)"].append(row.get("first_token_latency_s") or 0)
                qt_data["Latence inter-token (ms)"].append(row.get("inter_token_latency_ms") or 0)
                qt_data["Mémoire pic (Go)"].append(row.get("peak_memory_gb") or 0)
                qt_data["Chargement (s)"].append(row.get("model_load_time_s") or 0)
                qt_data["Stabilité"].append(row.get("stability") or "?")

            import pandas as pd
            df_qt = pd.DataFrame(qt_data)
            st.dataframe(df_qt, use_container_width=True, hide_index=True)

            # Graphiques comparatifs
            quant_labels = [row["quantization"] for row in table]
            quant_bits = [f'{row["bits"]}-bit' for row in table]

            col1, col2 = st.columns(2)

            with col1:
                # Tokens/s par quantification
                tps_vals = [row.get("tokens_per_second") or 0 for row in table]
                fig_tps = px.bar(
                    x=quant_labels, y=tps_vals,
                    labels={"x": "Quantification", "y": "Tokens/s"},
                    title=f"{model_name} — Débit d'inférence par quantification",
                    color=quant_labels,
                    color_discrete_sequence=px.colors.qualitative.Vivid,
                    text=tps_vals,
                )
                fig_tps.update_traces(texttemplate="%{text:.1f}", textposition="outside")
                fig_tps.update_layout(showlegend=False, xaxis_title="Quantification", yaxis_title="Tokens/s")
                st.plotly_chart(fig_tps, use_container_width=True)

            with col2:
                # Mémoire pic par quantification
                mem_vals = [row.get("peak_memory_gb") or 0 for row in table]
                fig_mem = px.bar(
                    x=quant_labels, y=mem_vals,
                    labels={"x": "Quantification", "y": "Mémoire pic (Go)"},
                    title=f"{model_name} — Mémoire pic par quantification",
                    color=quant_labels,
                    color_discrete_sequence=px.colors.qualitative.Safe,
                    text=[f"{v:.2f}" for v in mem_vals],
                )
                fig_mem.update_traces(textposition="outside")
                fig_mem.update_layout(showlegend=False, xaxis_title="Quantification", yaxis_title="Go")
                st.plotly_chart(fig_mem, use_container_width=True)

            col3, col4 = st.columns(2)

            with col3:
                # Latence premier token
                ftl_vals = [row.get("first_token_latency_s") or 0 for row in table]
                fig_ftl = px.bar(
                    x=quant_labels, y=ftl_vals,
                    labels={"x": "Quantification", "y": "Latence (s)"},
                    title=f"{model_name} — Latence du 1er token",
                    color=quant_labels,
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    text=[f"{v:.3f}" for v in ftl_vals],
                )
                fig_ftl.update_traces(textposition="outside")
                fig_ftl.update_layout(showlegend=False, xaxis_title="Quantification", yaxis_title="Secondes")
                st.plotly_chart(fig_ftl, use_container_width=True)

            with col4:
                # Temps de chargement
                load_vals = [row.get("model_load_time_s") or 0 for row in table]
                fig_load = px.bar(
                    x=quant_labels, y=load_vals,
                    labels={"x": "Quantification", "y": "Temps (s)"},
                    title=f"{model_name} — Temps de chargement du modèle",
                    color=quant_labels,
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    text=[f"{v:.1f}" for v in load_vals],
                )
                fig_load.update_traces(textposition="outside")
                fig_load.update_layout(showlegend=False, xaxis_title="Quantification", yaxis_title="Secondes")
                st.plotly_chart(fig_load, use_container_width=True)

            # Graphique combiné : taille fichier vs tokens/s (scatter)
            size_vals = [row.get("file_size_gb") or 0 for row in table]
            tps_vals = [row.get("tokens_per_second") or 0 for row in table]
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=size_vals,
                y=tps_vals,
                mode="markers+text",
                marker=dict(
                    size=[b * 5 + 10 for b in [row["bits"] for row in table]],
                    color=[row["bits"] for row in table],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Bits"),
                ),
                text=quant_labels,
                textposition="top center",
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Taille : %{x:.2f} Go<br>"
                    "Tokens/s : %{y:.1f}<br>"
                    "<extra></extra>"
                ),
            ))
            fig_scatter.update_layout(
                title=f"{model_name} — Compromis taille fichier vs performance",
                xaxis_title="Taille du fichier (Go)",
                yaxis_title="Tokens/s",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            # Latence inter-token si disponible
            itl_vals = [row.get("inter_token_latency_ms") or 0 for row in table]
            if any(v > 0 for v in itl_vals):
                fig_itl = px.bar(
                    x=quant_labels, y=itl_vals,
                    labels={"x": "Quantification", "y": "Latence (ms)"},
                    title=f"{model_name} — Latence inter-token moyenne",
                    color=quant_labels,
                    color_discrete_sequence=px.colors.qualitative.Set3,
                    text=[f"{v:.1f}" for v in itl_vals],
                )
                fig_itl.update_traces(textposition="outside")
                fig_itl.update_layout(showlegend=False)
                st.plotly_chart(fig_itl, use_container_width=True)

    # JSON brut
    with st.expander("Données brutes (JSON)"):
        st.json(data)


def _display_comparison(loaded_data: dict):
    """Affiche la comparaison entre plusieurs résultats avec couleurs distinctes."""
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd

    st.markdown("### Comparaison des résultats")

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
        for prefix in ["Apple ", "Intel(R) Core(TM) ", "AMD Ryzen "]:
            if prefix in cpu_short:
                cpu_short = cpu_short.split(prefix)[-1].split(" @")[0].split(" CPU")[0]
                break
        timestamp = data.get("timestamp", "")[:16].replace("T", " ")
        label = f"{cpu_short} ({timestamp})"
        if label in result_labels.values():
            label += f" #{idx+1}"
        result_labels[fname] = label
        result_colors[fname] = COLORS[idx % len(COLORS)]

    # ─── Légende couleurs ───
    legend_html = " &nbsp; ".join(
        f'<span style="display:inline-block;width:14px;height:14px;'
        f'background:{result_colors[f]};border-radius:3px;margin-right:4px;'
        f'vertical-align:middle;"></span>'
        f'<span style="vertical-align:middle;font-weight:600;">{result_labels[f]}</span>'
        for f in loaded_data
    )
    st.markdown(
        f'<div style="background:#f0f2f6;padding:10px 16px;border-radius:8px;'
        f'margin-bottom:20px;">'
        f'<b>🎨 Légende :</b> &nbsp; {legend_html}</div>',
        unsafe_allow_html=True,
    )

    # ─── Tableau comparatif matériel ───
    st.markdown("#### Comparaison matérielle")
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
    st.markdown("#### Comparaison CPU")
    has_cpu = False
    fig_cpu = go.Figure()
    for fname, data in loaded_data.items():
        classic = data.get("classic_benchmarks", {}).get("benchmarks", {})
        cpu_st = classic.get("cpu_single_thread", {}).get("results", {})
        cpu_mt = classic.get("cpu_multi_thread", {}).get("results", {})
        st_gflops = list(cpu_st.values())[-1].get("gflops", 0) if cpu_st else 0
        mt_gflops = list(cpu_mt.values())[-1].get("gflops", 0) if cpu_mt else 0
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
            r_bw = mem.get("read", {}).get("bandwidth_gb_s", 0)
            w_bw = mem.get("write", {}).get("bandwidth_gb_s", 0)
            c_bw = mem.get("copy", {}).get("bandwidth_gb_s", 0)
            fig_mem.add_trace(go.Bar(
                name=result_labels[fname],
                x=["Lecture", "Écriture", "Copie"],
                y=[r_bw, w_bw, c_bw],
                marker_color=result_colors[fname],
                text=[f"{r_bw:.1f}", f"{w_bw:.1f}", f"{c_bw:.1f}"],
                textposition="outside",
            ))
    if has_mem:
        st.markdown("#### Comparaison Mémoire")
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
        st.markdown("#### Comparaison GPU")
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
    st.markdown("#### Comparaison Inférence IA")

    all_models = {}
    for fname, data in loaded_data.items():
        ai_results = data.get("ai_benchmarks", {}).get("results", {})
        for model_key, model_data in ai_results.items():
            if model_data.get("summary"):
                all_models[model_key] = model_data.get("model", model_key)

    if all_models:
        model_keys_sorted = sorted(all_models.keys())
        model_names_sorted = [all_models[k] for k in model_keys_sorted]

        # Tokens/s
        fig_tps = go.Figure()
        fig_ftl = go.Figure()
        for fname, data in loaded_data.items():
            ai_results = data.get("ai_benchmarks", {}).get("results", {})
            tps_values = []
            ftl_values = []
            for model_key in model_keys_sorted:
                summary = ai_results.get(model_key, {}).get("summary", {})
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

        # Mémoire pic
        fig_mem_ai = go.Figure()
        for fname, data in loaded_data.items():
            ai_results = data.get("ai_benchmarks", {}).get("results", {})
            mem_values = []
            for model_key in model_keys_sorted:
                summary = ai_results.get(model_key, {}).get("summary", {})
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

        # Tableau comparatif
        st.markdown("#### Tableau comparatif complet")
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

    # ══════════════════════════════════════════════
    # Comparaison Quantification inter-machines
    # ══════════════════════════════════════════════
    all_quant_models = {}
    for fname, data in loaded_data.items():
        quant_comp = data.get("ai_benchmarks", {}).get("quantization_comparison", {})
        for model_key, comp_data in quant_comp.items():
            if model_key not in all_quant_models:
                all_quant_models[model_key] = comp_data.get("model_name", model_key)

    if all_quant_models:
        st.markdown("#### Comparaison Quantification")

        for model_key, model_name in all_quant_models.items():
            st.markdown(f"##### {model_name}")

            # Collecter toutes les quantifications testées
            all_quants = set()
            for fname, data in loaded_data.items():
                comp = data.get("ai_benchmarks", {}).get(
                    "quantization_comparison", {}
                ).get(model_key, {})
                for qk, qr in comp.get("results", {}).items():
                    if qr.get("summary"):
                        all_quants.add(qk)

            if not all_quants:
                continue

            quant_sorted = sorted(all_quants, key=lambda q: QUANTIZATION_VARIANTS.get(
                model_key, {}
            ).get(q, {}).get("bits", 0))

            # Tokens/s par quantification, par machine
            fig_q_tps = go.Figure()
            fig_q_mem = go.Figure()

            for fname, data in loaded_data.items():
                comp = data.get("ai_benchmarks", {}).get(
                    "quantization_comparison", {}
                ).get(model_key, {})
                results = comp.get("results", {})

                tps_vals = []
                mem_vals = []
                for qk in quant_sorted:
                    qr = results.get(qk, {})
                    summary = qr.get("summary", {})
                    tps_vals.append(summary.get("avg_tokens_per_second", 0))
                    mem_vals.append(summary.get("peak_memory_gb", 0))

                fig_q_tps.add_trace(go.Bar(
                    name=result_labels[fname],
                    x=quant_sorted,
                    y=tps_vals,
                    marker_color=result_colors[fname],
                    text=[f"{v:.1f}" if v > 0 else "" for v in tps_vals],
                    textposition="outside",
                ))
                fig_q_mem.add_trace(go.Bar(
                    name=result_labels[fname],
                    x=quant_sorted,
                    y=mem_vals,
                    marker_color=result_colors[fname],
                    text=[f"{v:.2f}" if v > 0 else "" for v in mem_vals],
                    textposition="outside",
                ))

            col1, col2 = st.columns(2)
            with col1:
                fig_q_tps.update_layout(
                    barmode="group",
                    title=f"{model_name} — Tokens/s par quantification",
                    yaxis_title="Tokens/s",
                    xaxis_title="Quantification",
                    legend_title="Résultat",
                )
                st.plotly_chart(fig_q_tps, use_container_width=True)
            with col2:
                fig_q_mem.update_layout(
                    barmode="group",
                    title=f"{model_name} — Mémoire pic par quantification",
                    yaxis_title="Go",
                    xaxis_title="Quantification",
                    legend_title="Résultat",
                )
                st.plotly_chart(fig_q_mem, use_container_width=True)

            # Tableau comparatif quantification multi-machine
            qt_table = {
                "Résultat": [], "Quantification": [], "Bits": [],
                "Taille (Go)": [], "Tokens/s": [],
                "1er token (s)": [], "Mémoire pic (Go)": [],
                "Chargement (s)": [],
            }
            for fname, data in loaded_data.items():
                comp = data.get("ai_benchmarks", {}).get(
                    "quantization_comparison", {}
                ).get(model_key, {})
                results = comp.get("results", {})
                for qk in quant_sorted:
                    qr = results.get(qk, {})
                    summary = qr.get("summary", {})
                    if summary:
                        qt_table["Résultat"].append(result_labels[fname])
                        qt_table["Quantification"].append(qk)
                        qt_table["Bits"].append(qr.get("bits", 0))
                        qt_table["Taille (Go)"].append(
                            round(qr.get("actual_file_size_gb", qr.get("file_size_gb", 0)), 3)
                        )
                        qt_table["Tokens/s"].append(summary.get("avg_tokens_per_second", 0))
                        qt_table["1er token (s)"].append(
                            summary.get("avg_first_token_latency_s", 0)
                        )
                        qt_table["Mémoire pic (Go)"].append(
                            summary.get("peak_memory_gb", 0)
                        )
                        qt_table["Chargement (s)"].append(
                            qr.get("model_load_time_s", 0)
                        )

            if qt_table["Résultat"]:
                st.dataframe(
                    pd.DataFrame(qt_table), use_container_width=True, hide_index=True
                )


# =============================================================================
# Router
# =============================================================================
if page == "Matériel":
    page_hardware()
elif page == "Benchmark":
    page_benchmark()
elif page == "Résultats":
    page_results()
