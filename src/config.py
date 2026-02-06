"""
ComputeLLM - Configuration et constantes globales.
Définit les modèles disponibles, les paramètres de benchmark et les chemins.
"""

import os
from pathlib import Path

# =============================================================================
# Chemins du projet
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Créer les répertoires s'ils n'existent pas
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# =============================================================================
# Modèles LLM disponibles pour le benchmark (GGUF via HuggingFace)
# =============================================================================
AVAILABLE_MODELS = {
    "tinyllama-1.1b": {
        "name": "TinyLlama 1.1B",
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "params": "1.1B",
        "size_gb": 0.7,
        "min_ram_gb": 2,
        "description": "Modèle ultra-léger, idéal pour tester tous les systèmes.",
    },
    "mistral-7b": {
        "name": "Mistral 7B",
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "params": "7B",
        "size_gb": 4.4,
        "min_ram_gb": 8,
        "description": "Modèle performant 7B paramètres, bon compromis taille/qualité.",
    },
    "llama2-13b": {
        "name": "Llama 2 13B",
        "repo_id": "TheBloke/Llama-2-13B-chat-GGUF",
        "filename": "llama-2-13b-chat.Q4_K_M.gguf",
        "params": "13B",
        "size_gb": 7.9,
        "min_ram_gb": 16,
        "description": "Modèle 13B paramètres, nécessite plus de mémoire.",
    },
    "codellama-34b": {
        "name": "CodeLlama 34B",
        "repo_id": "TheBloke/CodeLlama-34B-Instruct-GGUF",
        "filename": "codellama-34b-instruct.Q4_K_M.gguf",
        "params": "34B",
        "size_gb": 20.2,
        "min_ram_gb": 32,
        "description": "Modèle 34B paramètres, test des limites matérielles.",
    },
    "llama2-70b": {
        "name": "Llama 2 70B",
        "repo_id": "TheBloke/Llama-2-70B-chat-GGUF",
        "filename": "llama-2-70b-chat.Q4_K_M.gguf",
        "params": "70B",
        "size_gb": 40.5,
        "min_ram_gb": 64,
        "description": "Modèle 70B paramètres, réservé aux machines très puissantes.",
    },
}

# =============================================================================
# Prompt fixe pour les benchmarks d'inférence
# =============================================================================
BENCHMARK_PROMPT = (
    "Explain the concept of artificial intelligence in simple terms. "
    "What are its main applications and how does it impact our daily lives? "
    "Provide specific examples."
)

# =============================================================================
# Paramètres de benchmark
# =============================================================================
INFERENCE_CONFIG = {
    "max_tokens": 256,          # Nombre max de tokens générés
    "temperature": 0.7,         # Température de génération
    "top_p": 0.9,               # Top-p sampling
    "repeat_penalty": 1.1,      # Pénalité de répétition
    "n_ctx": 2048,              # Taille du contexte
    "seed": 42,                 # Graine pour reproductibilité
    "n_warmup_runs": 1,         # Nombre de runs d'échauffement
    "n_benchmark_runs": 3,      # Nombre de runs de benchmark
}

CLASSIC_BENCHMARK_CONFIG = {
    "matrix_sizes": [512, 1024, 2048],     # Tailles de matrices pour benchmark CPU
    "memory_test_size_mb": 256,            # Taille du test mémoire en Mo
    "n_iterations": 3,                     # Nombre d'itérations par test
}

# =============================================================================
# Paramètres de monitoring
# =============================================================================
MONITOR_INTERVAL = 0.5  # Intervalle d'échantillonnage en secondes
