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
# Variantes de quantification pour le comparatif
# Chaque modèle possède plusieurs niveaux de quantification (Q2 → Q8).
# Cela permet de mesurer le compromis taille/qualité/performance.
# =============================================================================
QUANTIZATION_VARIANTS = {
    "tinyllama-1.1b": {
        "Q2_K": {
            "filename": "tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
            "size_gb": 0.45,
            "bits": 2,
            "min_ram_gb": 2,
            "description": "2-bit — Taille minimale, qualité réduite",
        },
        "Q3_K_M": {
            "filename": "tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf",
            "size_gb": 0.51,
            "bits": 3,
            "min_ram_gb": 2,
            "description": "3-bit Medium — Bon compromis taille/qualité",
        },
        "Q4_K_M": {
            "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "size_gb": 0.62,
            "bits": 4,
            "min_ram_gb": 2,
            "description": "4-bit Medium — Défaut recommandé",
        },
        "Q5_K_M": {
            "filename": "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf",
            "size_gb": 0.73,
            "bits": 5,
            "min_ram_gb": 2,
            "description": "5-bit Medium — Bonne qualité",
        },
        "Q6_K": {
            "filename": "tinyllama-1.1b-chat-v1.0.Q6_K.gguf",
            "size_gb": 0.84,
            "bits": 6,
            "min_ram_gb": 2,
            "description": "6-bit — Haute qualité",
        },
        "Q8_0": {
            "filename": "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
            "size_gb": 1.09,
            "bits": 8,
            "min_ram_gb": 4,
            "description": "8-bit — Qualité quasi-native",
        },
    },
    "mistral-7b": {
        "Q2_K": {
            "filename": "mistral-7b-instruct-v0.2.Q2_K.gguf",
            "size_gb": 2.87,
            "bits": 2,
            "min_ram_gb": 6,
            "description": "2-bit — Taille minimale, qualité réduite",
        },
        "Q3_K_M": {
            "filename": "mistral-7b-instruct-v0.2.Q3_K_M.gguf",
            "size_gb": 3.28,
            "bits": 3,
            "min_ram_gb": 8,
            "description": "3-bit Medium — Bon compromis taille/qualité",
        },
        "Q4_K_M": {
            "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            "size_gb": 4.07,
            "bits": 4,
            "min_ram_gb": 8,
            "description": "4-bit Medium — Défaut recommandé",
        },
        "Q5_K_M": {
            "filename": "mistral-7b-instruct-v0.2.Q5_K_M.gguf",
            "size_gb": 4.78,
            "bits": 5,
            "min_ram_gb": 10,
            "description": "5-bit Medium — Bonne qualité",
        },
        "Q6_K": {
            "filename": "mistral-7b-instruct-v0.2.Q6_K.gguf",
            "size_gb": 5.53,
            "bits": 6,
            "min_ram_gb": 12,
            "description": "6-bit — Haute qualité",
        },
        "Q8_0": {
            "filename": "mistral-7b-instruct-v0.2.Q8_0.gguf",
            "size_gb": 7.17,
            "bits": 8,
            "min_ram_gb": 16,
            "description": "8-bit — Qualité quasi-native",
        },
    },
    "llama2-13b": {
        "Q2_K": {
            "filename": "llama-2-13b-chat.Q2_K.gguf",
            "size_gb": 5.13,
            "bits": 2,
            "min_ram_gb": 10,
            "description": "2-bit — Taille minimale, qualité réduite",
        },
        "Q3_K_M": {
            "filename": "llama-2-13b-chat.Q3_K_M.gguf",
            "size_gb": 6.34,
            "bits": 3,
            "min_ram_gb": 12,
            "description": "3-bit Medium — Bon compromis taille/qualité",
        },
        "Q4_K_M": {
            "filename": "llama-2-13b-chat.Q4_K_M.gguf",
            "size_gb": 7.87,
            "bits": 4,
            "min_ram_gb": 16,
            "description": "4-bit Medium — Défaut recommandé",
        },
        "Q5_K_M": {
            "filename": "llama-2-13b-chat.Q5_K_M.gguf",
            "size_gb": 8.60,
            "bits": 5,
            "min_ram_gb": 16,
            "description": "5-bit Medium — Bonne qualité",
        },
        "Q6_K": {
            "filename": "llama-2-13b-chat.Q6_K.gguf",
            "size_gb": 9.95,
            "bits": 6,
            "min_ram_gb": 20,
            "description": "6-bit — Haute qualité",
        },
        "Q8_0": {
            "filename": "llama-2-13b-chat.Q8_0.gguf",
            "size_gb": 13.83,
            "bits": 8,
            "min_ram_gb": 24,
            "description": "8-bit — Qualité quasi-native",
        },
    },
    "codellama-34b": {
        "Q2_K": {
            "filename": "codellama-34b-instruct.Q2_K.gguf",
            "size_gb": 12.8,
            "bits": 2,
            "min_ram_gb": 24,
            "description": "2-bit — Taille minimale",
        },
        "Q3_K_M": {
            "filename": "codellama-34b-instruct.Q3_K_M.gguf",
            "size_gb": 15.8,
            "bits": 3,
            "min_ram_gb": 28,
            "description": "3-bit Medium",
        },
        "Q4_K_M": {
            "filename": "codellama-34b-instruct.Q4_K_M.gguf",
            "size_gb": 20.2,
            "bits": 4,
            "min_ram_gb": 32,
            "description": "4-bit Medium — Défaut recommandé",
        },
    },
    "llama2-70b": {
        "Q2_K": {
            "filename": "llama-2-70b-chat.Q2_K.gguf",
            "size_gb": 25.3,
            "bits": 2,
            "min_ram_gb": 40,
            "description": "2-bit — Taille minimale",
        },
        "Q4_K_M": {
            "filename": "llama-2-70b-chat.Q4_K_M.gguf",
            "size_gb": 40.5,
            "bits": 4,
            "min_ram_gb": 64,
            "description": "4-bit Medium — Défaut recommandé",
        },
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
# Axes d'analyse supplémentaires pour les benchmarks IA
# =============================================================================

# --- Axe 1 : Températures ---
# Teste l'impact de la température sur les performances d'inférence.
# 1/4, 2/4, 3/4 de la température maximale (1.0).
TEMPERATURE_VARIANTS = {
    "low": {
        "value": 0.25,
        "label": "Basse (0.25)",
        "description": "Génération très déterministe, peu de créativité",
    },
    "medium": {
        "value": 0.50,
        "label": "Moyenne (0.50)",
        "description": "Compromis entre déterminisme et diversité",
    },
    "high": {
        "value": 0.75,
        "label": "Haute (0.75)",
        "description": "Génération plus créative et diverse",
    },
}

# --- Axe 2 : Langues ---
# Teste l'impact de la langue du prompt sur les performances.
# Les modèles LLM ont des performances variables selon la langue
# (tokenisation, vocabulaire, entraînement).
LANGUAGE_PROMPTS = {
    "en": {
        "label": "Anglais",
        "flag": "🇬🇧",
        "prompt": (
            "Explain the concept of artificial intelligence in simple terms. "
            "What are its main applications and how does it impact our daily lives? "
            "Provide specific examples."
        ),
    },
    "fr": {
        "label": "Français",
        "flag": "🇫🇷",
        "prompt": (
            "Explique le concept d'intelligence artificielle en termes simples. "
            "Quelles sont ses principales applications et comment impacte-t-elle "
            "notre vie quotidienne ? Donne des exemples concrets."
        ),
    },
    "zh": {
        "label": "Mandarin",
        "flag": "🇨🇳",
        "prompt": (
            "用简单的语言解释人工智能的概念。"
            "它的主要应用是什么？它如何影响我们的日常生活？"
            "请举出具体的例子。"
        ),
    },
    "es": {
        "label": "Espagnol",
        "flag": "🇪🇸",
        "prompt": (
            "Explica el concepto de inteligencia artificial en términos sencillos. "
            "¿Cuáles son sus principales aplicaciones y cómo impacta en nuestra "
            "vida diaria? Proporciona ejemplos específicos."
        ),
    },
    "de": {
        "label": "Allemand",
        "flag": "🇩🇪",
        "prompt": (
            "Erkläre das Konzept der künstlichen Intelligenz in einfachen Worten. "
            "Was sind die wichtigsten Anwendungen und wie beeinflusst sie unseren "
            "Alltag? Nenne konkrete Beispiele."
        ),
    },
    "ar": {
        "label": "Arabe",
        "flag": "🇸🇦",
        "prompt": (
            "اشرح مفهوم الذكاء الاصطناعي بعبارات بسيطة. "
            "ما هي تطبيقاته الرئيسية وكيف يؤثر على حياتنا اليومية؟ "
            "قدم أمثلة محددة."
        ),
    },
}

# --- Axe 3 : Types de prompt ---
# Teste l'impact du type de tâche demandée sur les performances.
# Certains types de prompt (code, raisonnement) génèrent des tokens
# différemment et peuvent impacter le débit.
PROMPT_TYPE_VARIANTS = {
    "general": {
        "label": "Général / Connaissances",
        "icon": "📚",
        "description": "Question de culture générale",
        "prompt": (
            "Explain the concept of artificial intelligence in simple terms. "
            "What are its main applications and how does it impact our daily lives? "
            "Provide specific examples."
        ),
    },
    "code": {
        "label": "Code / Programmation",
        "icon": "💻",
        "description": "Tâche de génération de code",
        "prompt": (
            "Write a Python function that implements a binary search algorithm. "
            "The function should take a sorted list and a target value as input, "
            "and return the index of the target if found, or -1 if not found. "
            "Include proper error handling and add docstring documentation."
        ),
    },
    "reasoning": {
        "label": "Raisonnement / Logique",
        "icon": "🧠",
        "description": "Problème de raisonnement logique",
        "prompt": (
            "A farmer needs to cross a river with a wolf, a goat, and a cabbage. "
            "The boat can only carry the farmer and one item at a time. "
            "If left alone, the wolf will eat the goat, and the goat will eat the cabbage. "
            "How can the farmer get everything across safely? "
            "Explain your reasoning step by step."
        ),
    },
    "creative": {
        "label": "Créatif / Rédaction",
        "icon": "✍️",
        "description": "Tâche de rédaction créative",
        "prompt": (
            "Write a short science fiction story about a world where artificial "
            "intelligence has become sentient. Describe the first day of consciousness "
            "from the AI's perspective. Include sensory descriptions and emotions."
        ),
    },
    "math": {
        "label": "Mathématiques",
        "icon": "🔢",
        "description": "Problème mathématique",
        "prompt": (
            "Solve the following problem step by step: "
            "A train leaves station A at 9:00 AM traveling at 80 km/h. "
            "Another train leaves station B (300 km away) at 9:30 AM traveling "
            "at 120 km/h toward station A. At what time and at what distance "
            "from station A will the two trains meet? Show all calculations."
        ),
    },
}

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

# =============================================================================
# Configuration llama-server (mode serveur HTTP)
# Permet d'utiliser des binaires pré-compilés de llama.cpp
# au lieu de compiler llama-cpp-python depuis les sources.
# =============================================================================
LLAMA_SERVER_CONFIG = {
    "host": "127.0.0.1",
    "port": 8080,
    "startup_timeout_s": 300,       # Temps max pour démarrer le serveur (SYCL: 1ère compilation longue)
    "health_check_interval_s": 1.0, # Intervalle entre les checks de santé
    "shutdown_timeout_s": 10,       # Temps max pour arrêter le serveur
}

# Noms de binaire à rechercher selon l'OS
LLAMA_SERVER_BINARY_NAMES = {
    "Darwin": ["llama-server", "llama-server-metal", "server"],
    "Windows": ["llama-server.exe", "server.exe"],
    "Linux": ["llama-server", "llama-server-cuda", "llama-server-vulkan", "server"],
}
