# 🖥️ ComputeLLM — AI Hardware Benchmark Tool

**Outil de benchmark multiplateforme dédié au Hardware IA, avec focus sur l'inférence locale de LLM.**

ComputeLLM permet de comparer les performances matérielles (CPU, GPU, RAM) de différentes machines lors de l'inférence locale de modèles de langage, en mettant en évidence les différences d'architecture :

| Architecture           | Exemple                     | Backend        |
| ---------------------- | --------------------------- | -------------- |
| x86 + GPU dédié NVIDIA | Intel/AMD CPU + NVIDIA RTX  | CUDA           |
| x86 + GPU dédié AMD    | Intel/AMD CPU + Radeon RX   | ROCm           |
| ARM + Mémoire unifiée  | Apple Silicon (M1/M2/M3/M4) | Metal          |
| CPU seul               | Tout processeur             | CPU (fallback) |

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

#### Linux (AMD GPU — ROCm / HIP)

```bash
CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install llama-cpp-python
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

#### Windows / Linux (NVIDIA — CUDA)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Linux (AMD GPU — ROCm)

Pour les GPU AMD (Radeon RX, Instinct), installer la version ROCm de PyTorch :

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

#### Windows / Linux (Intel GPU — XPU)

Pour les GPU Intel (Iris Xe, Arc), installer la version XPU de PyTorch :

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
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

## Pipeline détaillé du benchmark

Voici le déroulement complet d'un benchmark, étape par étape.

### Nettoyage système (`system_cleanup`)

Avant **chaque phase**, un nettoyage complet est effectué :

1. **Garbage collector Python** — 3 générations (`gc.collect(generation)` pour 0, 1, 2)
2. **Purge cache GPU** — `torch.cuda.empty_cache()` (NVIDIA), `torch.xpu.empty_cache()` (Intel), ou `torch.mps.empty_cache()` (Apple)
3. **Purge cache OS** — macOS : `sudo purge` ou `memory_pressure -l warn` · Linux : `drop_caches`
4. **Pause** — 1 seconde de stabilisation

### Monitoring en temps réel (`ResourceMonitor`)

Un thread de monitoring tourne en arrière-plan pendant chaque phase :

- **Fréquence** : 1 échantillon toutes les **0,5 s**
- **Métriques** : CPU % (`psutil`), RAM utilisée (Go), GPU % + VRAM + température (`nvidia-smi` ou `xpu-smi`)
- **Résumé** : moyenne, min, max CPU · pic RAM · pic GPU/VRAM/température

---

### Phase 1 — CPU Single-Thread

| Paramètre      | Valeur                                                                                                                  |
| -------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Outil          | `numpy.dot` (float32)                                                                                                   |
| Tailles        | 512×512, 1024×1024, 2048×2048                                                                                           |
| Itérations     | 3 par taille                                                                                                            |
| Thread forcing | `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1` |

**Calcul** : pour chaque taille $N$, on mesure le temps moyen sur 3 runs, puis :

$$\text{GFLOPS} = \frac{2 \times N^3}{\text{temps\_moyen} \times 10^9}$$

---

### Phase 2 — CPU Multi-Thread

| Paramètre  | Valeur                                          |
| ---------- | ----------------------------------------------- |
| Outil      | `numpy.dot` (float32)                           |
| Tailles    | 512×512, 1024×1024, 2048×2048                   |
| Itérations | 3 par taille                                    |
| Threads    | Tous les cœurs disponibles (via MKL / OpenBLAS) |

Même formule GFLOPS que la phase 1, mais avec tous les threads actifs.

---

### Phase 3 — Bande passante mémoire (RAM)

| Paramètre      | Valeur          |
| -------------- | --------------- |
| Outil          | NumPy (float64) |
| Taille du bloc | 256 Mo          |
| Itérations     | 3 par opération |

Trois opérations sont mesurées :

| Opération | Code                  | Description                 |
| --------- | --------------------- | --------------------------- |
| Écriture  | `np.ones(n, float64)` | Allocation et remplissage   |
| Lecture   | `np.sum(a)`           | Parcours complet du tableau |
| Copie     | `a.copy()`            | Duplication en mémoire      |

**Calcul** :

$$\text{Bande passante (Go/s)} = \frac{\text{taille\_données\_Go}}{\text{temps\_moyen}}$$

---

### Phase 4 — GPU Compute

| Paramètre | Valeur                                         |
| --------- | ---------------------------------------------- |
| Outil     | `torch.mm` (PyTorch, float32)                  |
| Tailles   | 1024×1024, 2048×2048, 4096×4096                |
| Warmup    | 1 run (non comptabilisé)                       |
| Runs      | 3 par taille                                   |
| Backends  | CUDA (NVIDIA) · SYCL/XPU (Intel) · MPS (Apple) |

**Déroulement** :

1. Détection automatique du backend GPU (priorité : CUDA → XPU/IPEX → MPS)
2. Warmup : 1 multiplication non chronométrée pour initialiser le GPU
3. Pour chaque taille, 3 runs chronométrés avec synchronisation (`torch.cuda.synchronize()` / `torch.xpu.synchronize()` / `torch.mps.synchronize()`)
4. Calcul GFLOPS identique aux phases CPU

**Protection Level Zero** : si le driver Intel crashe (`UR_RESULT_ERROR_UNKNOWN`), le benchmark GPU est ignoré proprement avec un message d'aide.

Si aucun GPU compatible n'est détecté, la phase est marquée « Ignoré » avec un conseil contextuel (ex : « Vous avez un GPU Intel mais PyTorch est compilé pour CUDA »).

---

### Phase 5 — Inférence IA (modèles LLM)

Pour **chaque modèle sélectionné** (ex : TinyLlama 1.1B, Mistral 7B…) :

#### 5.1 — Pré-vérifications

- **RAM disponible** : si la RAM totale < RAM minimale requise par le modèle → skip
- **Téléchargement** : si le fichier GGUF n'est pas déjà présent dans `models/`, il est téléchargé automatiquement depuis Hugging Face

#### 5.2 — Détection du backend

Ordre de priorité :

1. **NVIDIA** → CUDA (`n_gpu_layers = -1`, toutes les couches sur GPU)
2. **Intel** → SYCL (`n_gpu_layers = -1`, via `_detect_intel_gpu()` 4 méthodes)
3. **Apple** → Metal (`n_gpu_layers = -1`)
4. **Aucun** → CPU uniquement (`n_gpu_layers = 0`)

#### 5.3 — Chargement du modèle

- Bibliothèque : `llama-cpp-python` (mode natif) ou `llama-server` (mode serveur HTTP)
- Contexte : `n_ctx = 2048` tokens
- Seed : `42` (reproductibilité)
- Le temps de chargement est mesuré (`model_load_time_s`)

#### 5.4 — Échauffement (warmup)

- **1 run** avec `max_tokens = 32` (résultat ignoré)
- But : initialiser les caches KV et le runtime GPU

#### 5.5 — Runs de benchmark

- **3 runs** consécutifs, chacun mesuré individuellement
- Le `ResourceMonitor` tourne en arrière-plan pendant les 3 runs

**Paramètres d'inférence** (identiques pour chaque run) :

| Paramètre        | Valeur                                                             |
| ---------------- | ------------------------------------------------------------------ |
| Prompt           | `"Explain the concept of artificial intelligence in simple terms"` |
| Format           | Chat completion (messages system + user)                           |
| `max_tokens`     | 256                                                                |
| `temperature`    | 0.7                                                                |
| `top_p`          | 0.9                                                                |
| `repeat_penalty` | 1.1                                                                |
| `seed`           | 42                                                                 |

#### 5.6 — Mesures par run

Chaque run est en mode **streaming** (token par token). Les métriques mesurées :

| Métrique                     | Méthode de mesure                                                          |
| ---------------------------- | -------------------------------------------------------------------------- |
| **Latence 1er token** (s)    | `time.perf_counter()` entre le début et la réception du 1er token non-vide |
| **Tokens/s**                 | `tokens_générés / temps_total`                                             |
| **Latence inter-token** (ms) | Moyenne des deltas `time.perf_counter()` entre tokens consécutifs (× 1000) |
| **P90 inter-token** (ms)     | 90e percentile des deltas inter-token                                      |
| **Mémoire avant/après** (Go) | `psutil.Process().memory_info().rss` converti en Go                        |

#### 5.7 — Agrégation des résultats

Sur les **3 runs** (ou ceux ayant réussi) :

| Statistique                 | Calcul                                         |
| --------------------------- | ---------------------------------------------- |
| `avg_tokens_per_second`     | Moyenne des tokens/s des runs réussis          |
| `std_tokens_per_second`     | Écart-type des tokens/s                        |
| `avg_first_token_latency_s` | Moyenne des latences 1er token                 |
| `peak_memory_gb`            | Maximum de `memory_after_gb` sur tous les runs |
| `stability`                 | `"stable"` si 3/3 réussis, `"unstable"` sinon  |

---

### Phase 6 — Comparaison des quantifications

Pour chaque modèle sélectionné, si plusieurs quantifications sont disponibles (ex : Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0) :

1. **Nettoyage système** entre chaque variante
2. Même pipeline que la phase 5 (download → load → warmup → 3 runs)
3. Mesures additionnelles par quantification :

| Métrique               | Description                                    |
| ---------------------- | ---------------------------------------------- |
| `actual_file_size_gb`  | Taille réelle du fichier GGUF sur disque       |
| `model_load_time_s`    | Temps de chargement du modèle                  |
| `memory_after_load_gb` | Mémoire RSS après chargement (avant inférence) |

**Tableau comparatif** généré automatiquement avec : tokens/s, latence 1er token, latence inter-token, mémoire pic, temps de chargement, stabilité — pour chaque quantification.

---

### Phase 7 — Sauvegarde des résultats

- **Format** : JSON structuré dans `results/benchmark_{CPU}_{OS}_{timestamp}.json`
- **Contenu** : hardware détecté, résultats classiques, résultats IA, monitoring, comparaisons de quantification
- **Export** : CSV disponible depuis l'interface Streamlit

---

### Phase 8 — Comparaison de températures 🌡️

Pour chaque modèle sélectionné, teste l'impact de la température sur les performances d'inférence.

**Variants testés** :
| Clé | Température | Description |
| -------- | ----------- | -------------------------------- |
| `low` | 0.25 | Réponses déterministes, précises |
| `medium` | 0.50 | Équilibre précision/créativité |
| `high` | 0.75 | Réponses plus créatives/variées |

**Méthodologie** :

1. Chargement du modèle une seule fois (ou démarrage du serveur)
2. Phase de chauffe (1 run)
3. Pour chaque température : 3 runs de benchmark avec le même prompt, seule la température change
4. Agrégation : moyenne et écart-type de tokens/s, first-token latency, inter-token latency, mémoire pic, stabilité

**Tableau comparatif** : tokens/s, latence inter-token, mémoire pic — par variante de température. Graphiques Plotly dans l'interface.

---

### Phase 9 — Comparaison multilingue 🌍

Évalue si la langue du prompt impacte les performances d'inférence.

**Langues testées** :
| Clé | Langue | Drapeau |
| ---- | -------- | ------- |
| `en` | Anglais | 🇬🇧 |
| `fr` | Français | 🇫🇷 |
| `zh` | Mandarin | 🇨🇳 |
| `es` | Espagnol | 🇪🇸 |
| `de` | Allemand | 🇩🇪 |
| `ar` | Arabe | 🇸🇦 |

**Méthodologie** :

1. Chargement du modèle une seule fois
2. Phase de chauffe
3. Pour chaque langue : 3 runs avec la même question traduite dans la langue cible (température fixe)
4. Agrégation identique à la Phase 8

**Objectif** : Détecter si la tokenisation de certaines langues (chinois, arabe) impacte le débit en tokens/s ou la latence.

---

### Phase 10 — Comparaison par type de prompt 📝

Mesure l'impact du type de tâche demandée sur les performances d'inférence.

**Types de prompt testés** :
| Clé | Type | Icône | Description |
| ----------- | ---------- | ----- | ----------------------------------- |
| `general` | Général | 💬 | Question de culture générale |
| `code` | Code | 💻 | Génération de fonction Python |
| `reasoning` | Réflexion | 🧠 | Raisonnement logique / puzzle |
| `creative` | Créatif | 🎨 | Écriture créative (poème, histoire) |
| `math` | Maths | 🔢 | Résolution de problème mathématique |

**Méthodologie** :

1. Chargement du modèle une seule fois
2. Phase de chauffe
3. Pour chaque type : 3 runs avec un prompt dédié au type de tâche (température fixe)
4. Agrégation identique aux phases précédentes

**Objectif** : Identifier si certains types de prompts (code vs créatif) provoquent des différences de performance significatives (longueur de génération, patterns de tokens).

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

- [x] **Détection matérielle** — ✅ Support AMD complet
  - ~~Ajouter la détection des GPU AMD~~ ✔️
    - `rocm-smi` (Linux) ✔️
    - `lspci` (fallback Linux) ✔️
    - WMI / Win32_VideoController (Windows) ✔️
    - PyTorch ROCm (`torch.version.hip`) ✔️
  - ~~Ajouter la détection des GPU Intel (Arc / XPU)~~ ✔️
  - ~~Ajouter les backends détectés : `rocm`, `xpu`, `sycl`~~ ✔️

- [x] **Benchmark classique GPU** — ✅ Support AMD complet
  - ~~Support explicite AMD ROCm (identifier via `torch.version.hip`)~~ ✔️
  - ~~Support Intel XPU (`torch.xpu.is_available()`)~~ ✔️
  - ~~Synchronisation adaptée par device~~ ✔️ (ROCm utilise `torch.cuda.synchronize()`)
  - ~~Ajouter monitoring AMD : `rocm-smi --showuse --showmemuse --showtemp`~~ ✔️
  - ~~Distinguer CUDA vs ROCm dans l’affichage des résultats~~ ✔️

- [ ] **Benchmark AI / LLM**
  - Étendre `detect_best_backend()` :
    - ~~ROCm (HIPBLAS)~~ ✔️
    - Vulkan
    - ~~SYCL (Intel)~~ ✔️
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
  - ~~Agréger les métriques NVIDIA / AMD / Intel dans `ResourceMonitor`~~ ✔️
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
