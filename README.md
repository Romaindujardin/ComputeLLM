# 🖥️ ComputeLLM — AI Hardware Benchmark Tool

**Outil de benchmark multiplateforme dédié au Hardware IA, avec focus sur l'inférence locale de LLM.**

ComputeLLM permet de comparer les performances matérielles (CPU, GPU, RAM) de différentes machines lors de l'inférence locale de modèles de langage, en mettant en évidence les différences d'architecture :

| Architecture          | Exemple                     | Backend        |
| --------------------- | --------------------------- | -------------- |
| x86 + GPU dédié       | Intel/AMD + NVIDIA RTX      | CUDA           |
| ARM + Mémoire unifiée | Apple Silicon (M1/M2/M3/M4) | Metal          |
| CPU seul              | Tout processeur             | CPU (fallback) |

---

## Fonctionnalités

- **Détection matérielle automatique** : OS, CPU (modèle, cœurs, fréquence), GPU (VRAM, backend), RAM (totale, disponible, unifiée)
- **Benchmarks classiques** : CPU single-thread, CPU multi-thread, bande passante mémoire, GPU compute
- **Benchmarks IA** : Inférence locale de LLM via `llama-cpp-python` (GGUF) ou `llama-server` (HTTP)
- **Mode llama-server** : Utilise des binaires pré-compilés — aucune compilation requise côté Python
- **Modèles supportés** : TinyLlama 1.1B, Mistral 7B, Llama 2 13B, CodeLlama 34B, Llama 2 70B
- **Téléchargement automatique** depuis Hugging Face
- **Métriques mesurées** : tokens/s, latence du 1er token, mémoire utilisée, stabilité
- **Interface graphique** Streamlit avec bouton unique de lancement
- **Comparaison multi-machines** avec graphiques interactifs
- **Export** JSON et CSV

---

## Architecture

```
ComputeLLM/
├── main.py                    # Point d'entrée (CLI + GUI)
├── app.py                     # Application Streamlit (GUI)
├── requirements.txt           # Dépendances Python
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration et constantes
│   ├── hardware_detect.py     # Détection matérielle
│   ├── benchmark_classic.py   # Benchmarks CPU/GPU/RAM
│   ├── benchmark_ai.py        # Benchmarks inférence LLM
│   ├── llama_server.py        # Gestionnaire llama-server (HTTP)
│   └── results_manager.py     # Sauvegarde et comparaison
├── models/                    # Modèles GGUF téléchargés
└── results/                   # Résultats de benchmark (JSON)
```

---

## Installation

### Prérequis

- Python 3.10 ou supérieur
- pip

### 1. Cloner le dépôt

```bash
git clone https://github.com/Romaindujardin/ComputeLLM.git
cd ComputeLLM
```

### 2. Créer un environnement virtuel

```bash
python3 -m venv .venv
```

## Activer l'environnement - MacOS

```bash
source .venv/bin/activate
```

## Activer l'environnement - Windows

```bash
.venv\Scripts\activate
```

### 2. Installer les dépendances de base

```bash
pip install --upgrade pip setuptools wheel
```

```bash
pip install -r requirements.txt
```

### 3. Installer llama-cpp-python (selon votre matériel)

> **Alternative sans compilation** : voir la section [Mode llama-server](#mode-llama-server-sans-compilation) ci-dessous.

#### macOS (Apple Silicon — Metal)

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```

#### Windows / Linux (NVIDIA GPU — CUDA)

```bash
set CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

#### CPU uniquement (fallback)

```bash
pip install llama-cpp-python
```

### 4. (Optionnel) Installer PyTorch pour les benchmarks GPU classiques

#### macOS

```bash
pip install torch torchvision
```

#### Windows / Linux (CUDA)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 5. Mode llama-server (sans compilation)

Si vous ne pouvez pas (ou ne voulez pas) compiler `llama-cpp-python` depuis les sources (problèmes de Visual Studio, oneAPI Toolkit, etc.), vous pouvez utiliser le **mode llama-server** :

1. **Télécharger le binaire pré-compilé** depuis les [releases de llama.cpp](https://github.com/ggerganov/llama.cpp/releases) :
   - Windows CUDA : `llama-*-bin-win-cuda-*`
   - Windows Vulkan : `llama-*-bin-win-vulkan-*`
   - Windows SYCL : `llama-*-bin-win-sycl-*`
   - Linux CUDA : `llama-*-bin-ubuntu-*-cuda-*`
   - macOS Metal : `llama-*-bin-macos-*`

2. **Extraire l'archive** et repérer le binaire `llama-server` (ou `llama-server.exe` sur Windows)

3. **Dans ComputeLLM**, activer le toggle **"Utiliser llama-server"** dans la page Benchmark, puis configurer :
   - **Mode Auto** : indiquer le chemin vers le binaire `llama-server`. ComputeLLM démarre/arrête le serveur automatiquement.
   - **Mode Manuel** : lancer `llama-server` vous-même, puis renseigner l'adresse (ex: `127.0.0.1:8080`).

#### Lancement manuel de llama-server

```bash
# Exemple : lancer le serveur avec un modèle GGUF
./llama-server -m models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --port 8080 -ngl -1
```

> Le serveur expose une API compatible OpenAI sur `http://127.0.0.1:8080`. ComputeLLM s'y connecte automatiquement pour les benchmarks.

---

## Utilisation

### Interface graphique (recommandé)

```bash
python main.py --gui
```

ou simplement :

```bash
python main.py
```

L'interface Streamlit s'ouvre dans votre navigateur avec 3 pages :

1. **Matériel** — Détection et affichage de la configuration
2. **Benchmark** — Lancement avec un seul bouton
3. **Résultats** — Visualisation, comparaison et export

### Ligne de commande (CLI)

```bash
# Tous les benchmarks
python main.py --cli

# Modèle spécifique
python main.py --cli --models tinyllama-1.1b mistral-7b

# Benchmarks classiques uniquement
python main.py --cli --skip-ai

# Benchmarks IA uniquement
python main.py --cli --skip-classic

# Détecter le matériel uniquement
python main.py --detect
```

---

## Modèles disponibles

| Clé              | Modèle                 | Paramètres | Taille (Q4_K_M) | RAM min |
| ---------------- | ---------------------- | ---------- | --------------- | ------- |
| `tinyllama-1.1b` | TinyLlama 1.1B Chat    | 1.1B       | 0.7 Go          | 2 Go    |
| `mistral-7b`     | Mistral 7B Instruct    | 7B         | 4.4 Go          | 8 Go    |
| `llama2-13b`     | Llama 2 13B Chat       | 13B        | 7.9 Go          | 16 Go   |
| `codellama-34b`  | CodeLlama 34B Instruct | 34B        | 20.2 Go         | 32 Go   |
| `llama2-70b`     | Llama 2 70B Chat       | 70B        | 40.5 Go         | 64 Go   |

Les modèles sont téléchargés automatiquement depuis Hugging Face lors du premier benchmark.

---

## Métriques mesurées

### Benchmarks classiques

| Métrique              | Description                      |
| --------------------- | -------------------------------- |
| GFLOPS (ST)           | Performance CPU single-thread    |
| GFLOPS (MT)           | Performance CPU multi-thread     |
| Bande passante (Go/s) | Lecture, écriture, copie mémoire |
| GFLOPS GPU            | Performance GPU (CUDA/Metal)     |

### Benchmarks IA

| Métrique                 | Description                    |
| ------------------------ | ------------------------------ |
| Tokens/s                 | Débit d'inférence              |
| Latence 1er token (s)    | Temps avant le premier token   |
| Latence inter-token (ms) | Temps moyen entre chaque token |
| Mémoire pic (Go)         | Mémoire maximale utilisée      |
| Stabilité                | Nombre de runs réussis / total |

### Monitoring en temps réel

| Métrique        | Description                    |
| --------------- | ------------------------------ |
| CPU %           | Utilisation CPU (moyenne, max) |
| RAM Go          | Utilisation RAM (pic)          |
| GPU %           | Utilisation GPU (si NVIDIA)    |
| Température GPU | Si disponible                  |

---

## Exemple de résultat (JSON)

```json
{
  "id": "20260206_143022",
  "timestamp": "2026-02-06T14:30:22",
  "hardware": {
    "os": { "system": "Darwin", "machine": "arm64" },
    "cpu": { "model": "Apple M2 Pro", "physical_cores": 12 },
    "gpu": { "primary_backend": "metal" },
    "ram": { "total_gb": 32.0, "unified_memory": true }
  },
  "classic_benchmarks": {
    "benchmarks": {
      "cpu_multi_thread": {
        "results": { "2048x2048": { "gflops": 312.5 } }
      }
    }
  },
  "ai_benchmarks": {
    "results": {
      "tinyllama-1.1b": {
        "summary": {
          "avg_tokens_per_second": 85.3,
          "avg_first_token_latency_s": 0.12,
          "stability": "stable"
        }
      }
    }
  }
}
```

---

## Configuration avancée

Modifiez `src/config.py` pour ajuster :

- **Modèles** : Ajouter/supprimer des modèles GGUF
- **Prompt** : Modifier le prompt de benchmark
- **Paramètres d'inférence** : `max_tokens`, `temperature`, nombre de runs
- **Tailles de matrices** : Pour les benchmarks CPU
- **Intervalle de monitoring** : Fréquence d'échantillonnage

---

## To-Do

### Haute priorité

- [ ] **Détection matérielle**
  - Ajouter la détection des GPU AMD :
    - `rocm-smi` (Linux)
    - `lspci` (fallback Linux)
    - WMI / Win32_VideoController (Windows)
  - Ajouter la détection des GPU Intel (Arc / XPU) :
    - `xpu-smi`
    - Level Zero
    - `lspci` (Linux)
    - WMI (Windows)
  - Ajouter les backends détectés : `rocm`, `xpu`, `sycl`

- [ ] **Benchmark classique GPU**
  - Support explicite AMD ROCm (identifier via `torch.version.hip`)
  - Support Intel XPU (`torch.xpu.is_available()`)
  - Synchronisation adaptée par device :
    - `torch.cuda.synchronize()`
    - `torch.xpu.synchronize()`
    - `torch.mps.synchronize()`
  - Ajouter monitoring :
    - AMD : `rocm-smi --showuse --showmemuse --showtemp`
    - Intel : `xpu-smi dump` / `intel_gpu_top`
  - Distinguer CUDA vs ROCm dans l’affichage des résultats

- [ ] **Benchmark AI / LLM**
  - Étendre `detect_best_backend()` :
    - ROCm (HIPBLAS)
    - Vulkan
    - SYCL (Intel)
    - CLBlast / OpenCL (fallback générique)
  - Gérer explicitement les backends llama-cpp-python :
    - `-DGGML_CUDA=on`
    - `-DGGML_HIPBLAS=on`
    - `-DGGML_VULKAN=on`
    - `-DGGML_SYCL=on`
    - `-DGGML_CLBLAST=on`
  - Vérifier à l’exécution que le backend compilé correspond au GPU détecté - ~~Mode llama-server (binaires pré-compilés, zéro compilation)~~ ✅

---

### Priorité moyenne

- [ ] **Installation guidée de llama-cpp-python**
  - Détecter automatiquement le GPU au premier lancement
  - Proposer la bonne commande d’installation selon la plateforme
  - Ajouter un script d’installation automatique par OS
  - Afficher un avertissement si version CPU-only détectée - ~~Mode llama-server comme alternative sans compilation~~ ✅
- [ ] **Monitoring unifié**
  - Agréger les métriques NVIDIA / AMD / Intel dans `ResourceMonitor`
  - Normaliser le format des métriques (utilisation %, VRAM, température)

---

### Priorité basse

- [ ] **Support multi-GPU**
  - Détection de plusieurs GPU
  - Sélection via :
    - `CUDA_VISIBLE_DEVICES`
    - `HIP_VISIBLE_DEVICES`
    - `ZE_AFFINITY_MASK`
  - Benchmark individuel par GPU

- [ ] **Gestion RAM vs VRAM**
  - Distinguer RAM système et VRAM GPU
  - Adapter `get_compatible_quantizations()` en fonction de la VRAM
  - Recommandations dynamiques de modèles selon mémoire disponible

- [ ] **Backend Vulkan universel**
  - Ajouter détection via `vulkaninfo`
  - Documenter Vulkan comme backend cross-vendor (AMD / Intel / NVIDIA)

- [ ] **Documentation**
  - Ajouter tableau comparatif des backends supportés par plateforme
  - Documenter les prérequis ROCm / SYCL / Vulkan
  - Ajouter exemples d’installation par architecture :
    - Windows x86 + NVIDIA
    - Linux + AMD ROCm
    - Windows/Linux + Vulkan
    - macOS ARM + Metal

---

## Notes importantes

- Les modèles GGUF sont téléchargés dans le dossier `models/` (plusieurs Go par modèle).
- L'inférence de gros modèles (34B, 70B) nécessite beaucoup de RAM.
- Sur Apple Silicon, la mémoire unifiée est partagée entre CPU et GPU.
- Les résultats varient selon la charge système et la température du processeur.
- Pour des résultats reproductibles, fermez les applications gourmandes en ressources.
