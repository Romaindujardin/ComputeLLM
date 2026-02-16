"""
ComputeLLM - Module de gestion du serveur llama.cpp (llama-server).

Permet d'utiliser des binaires pré-compilés de llama.cpp au lieu de
compiler llama-cpp-python depuis les sources. Le serveur expose une API
HTTP compatible OpenAI, ce qui permet le benchmark sans aucune compilation.

Deux modes :
  - Auto : ComputeLLM démarre/arrête llama-server automatiquement
  - Manuel : l'utilisateur lance llama-server séparément, ComputeLLM se connecte
"""

import json
import os
import platform
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import psutil

from src.config import (
    BENCHMARK_PROMPT,
    INFERENCE_CONFIG,
    LLAMA_SERVER_BINARY_NAMES,
    LLAMA_SERVER_CONFIG,
    MODELS_DIR,
)


# =============================================================================
# Recherche du binaire llama-server
# =============================================================================

def find_llama_server_binary(custom_path: Optional[str] = None) -> Optional[str]:
    """
    Recherche le binaire llama-server sur le système.

    Ordre de recherche :
      1. Chemin personnalisé (si fourni)
      2. Variable d'environnement LLAMA_SERVER_PATH
      3. PATH système (shutil.which)
      4. Répertoires courants du projet (./bin/, ./llama.cpp/build/)

    Returns:
        Chemin absolu vers le binaire, ou None si introuvable.
    """
    system = platform.system()

    # 1. Chemin personnalisé
    if custom_path:
        p = Path(custom_path)
        if p.is_file() and os.access(str(p), os.X_OK):
            return str(p.resolve())
        # Si c'est un dossier, chercher dedans
        if p.is_dir():
            for name in LLAMA_SERVER_BINARY_NAMES.get(system, ["llama-server"]):
                candidate = p / name
                if candidate.is_file() and os.access(str(candidate), os.X_OK):
                    return str(candidate.resolve())

    # 2. Variable d'environnement
    env_path = os.environ.get("LLAMA_SERVER_PATH")
    if env_path:
        p = Path(env_path)
        if p.is_file() and os.access(str(p), os.X_OK):
            return str(p.resolve())

    # 3. PATH système
    for name in LLAMA_SERVER_BINARY_NAMES.get(system, ["llama-server"]):
        found = shutil.which(name)
        if found:
            return str(Path(found).resolve())

    # 4. Répertoires locaux courants
    project_root = Path(__file__).parent.parent
    local_dirs = [
        project_root / "bin",
        project_root / "llama.cpp" / "build" / "bin",
        project_root / "llama-server",
    ]
    for d in local_dirs:
        if d.is_dir():
            for name in LLAMA_SERVER_BINARY_NAMES.get(system, ["llama-server"]):
                candidate = d / name
                if candidate.is_file() and os.access(str(candidate), os.X_OK):
                    return str(candidate.resolve())

    return None


# =============================================================================
# Gestionnaire du serveur llama-server
# =============================================================================

class LlamaServerManager:
    """
    Gère le cycle de vie d'un processus llama-server.
    Peut démarrer, arrêter et vérifier l'état du serveur.
    """

    def __init__(
        self,
        binary_path: Optional[str] = None,
        host: str = None,
        port: int = None,
    ):
        self.binary_path = binary_path
        self.host = host or LLAMA_SERVER_CONFIG["host"]
        self.port = port or LLAMA_SERVER_CONFIG["port"]
        self._process: Optional[subprocess.Popen] = None
        self._model_path: Optional[str] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def is_running(self) -> bool:
        """Vérifie si le serveur est accessible via son endpoint /health."""
        try:
            import requests
            resp = requests.get(f"{self.base_url}/health", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def get_server_info(self) -> Optional[Dict[str, Any]]:
        """Récupère les informations du serveur (modèle chargé, etc.)."""
        try:
            import requests
            # Essayer /props (llama-server >= b2500)
            resp = requests.get(f"{self.base_url}/props", timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass

        # Fallback : juste le status
        if self.is_running():
            return {"status": "running", "url": self.base_url}
        return None

    def start(
        self,
        model_path: str,
        n_gpu_layers: int = -1,
        n_ctx: int = 2048,
        extra_args: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> bool:
        """
        Démarre le serveur llama-server avec le modèle spécifié.

        Args:
            model_path: Chemin vers le fichier GGUF.
            n_gpu_layers: Nombre de couches GPU (-1 = toutes).
            n_ctx: Taille du contexte.
            extra_args: Arguments CLI supplémentaires.
            progress_callback: Callback(progress, message).

        Returns:
            True si le serveur a démarré avec succès.
        """
        if self._process is not None:
            self.stop()

        if not self.binary_path:
            raise FileNotFoundError(
                "Binaire llama-server introuvable. "
                "Téléchargez-le depuis https://github.com/ggerganov/llama.cpp/releases"
            )

        if not Path(self.binary_path).is_file():
            raise FileNotFoundError(f"Binaire introuvable : {self.binary_path}")

        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Modèle introuvable : {model_path}")

        self._model_path = model_path

        # Construire la commande
        cmd = [
            self.binary_path,
            "-m", str(model_path),
            "--host", self.host,
            "--port", str(self.port),
            "-ngl", str(n_gpu_layers),
            "-c", str(n_ctx),
            "--seed", str(INFERENCE_CONFIG.get("seed", 42)),
        ]
        if extra_args:
            cmd.extend(extra_args)

        if progress_callback:
            progress_callback(0.0, f"Démarrage llama-server sur {self.host}:{self.port}...")

        # Démarrer le processus
        try:
            # Rediriger stdout/stderr pour ne pas polluer le terminal
            kwargs = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
            }
            if platform.system() == "Windows":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["preexec_fn"] = os.setsid

            self._process = subprocess.Popen(cmd, **kwargs)
        except PermissionError:
            raise PermissionError(
                f"Permission refusée pour exécuter {self.binary_path}. "
                f"Sur macOS/Linux : chmod +x {self.binary_path}"
            )
        except Exception as e:
            raise RuntimeError(f"Erreur lancement llama-server : {e}")

        # Attendre que le serveur soit prêt
        timeout = LLAMA_SERVER_CONFIG["startup_timeout_s"]
        interval = LLAMA_SERVER_CONFIG["health_check_interval_s"]
        elapsed = 0.0

        while elapsed < timeout:
            # Vérifier que le process est toujours vivant
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"llama-server s'est arrêté prématurément (code {self._process.returncode}).\n"
                    f"Erreur : {stderr[:500]}"
                )

            if self.is_running():
                if progress_callback:
                    progress_callback(1.0, f"llama-server prêt ({elapsed:.0f}s)")
                return True

            time.sleep(interval)
            elapsed += interval

            if progress_callback:
                p = min(elapsed / timeout, 0.95)
                progress_callback(p, f"Attente llama-server... ({elapsed:.0f}s/{timeout}s)")

        # Timeout
        self.stop()
        raise TimeoutError(
            f"llama-server n'a pas démarré en {timeout}s. "
            "Le chargement du modèle peut prendre du temps pour les gros modèles."
        )

    def stop(self):
        """Arrête le serveur llama-server."""
        if self._process is None:
            return

        try:
            if platform.system() == "Windows":
                # Sur Windows, utiliser taskkill pour tuer l'arbre de processus
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(self._process.pid)],
                        capture_output=True, timeout=LLAMA_SERVER_CONFIG["shutdown_timeout_s"],
                    )
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    self._process.terminate()
            else:
                # Envoyer SIGTERM au group de processus
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)

            try:
                self._process.wait(timeout=LLAMA_SERVER_CONFIG["shutdown_timeout_s"])
            except subprocess.TimeoutExpired:
                if platform.system() == "Windows":
                    self._process.kill()
                else:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                self._process.wait(timeout=5)
        except (ProcessLookupError, OSError):
            pass  # Processus déjà terminé
        finally:
            self._process = None
            self._model_path = None

    def __del__(self):
        self.stop()


# =============================================================================
# Utilitaires internes
# =============================================================================

def _find_llama_server_pid() -> Optional[int]:
    """
    Recherche le PID du processus llama-server en cours d'exécution.
    Utilisé pour mesurer la mémoire réelle consommée par le serveur.
    """
    try:
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                name = proc.info.get("name", "").lower()
                if "llama-server" in name or "llama_server" in name:
                    return proc.info["pid"]
                # Vérifier aussi dans la ligne de commande
                cmdline = proc.info.get("cmdline") or []
                for arg in cmdline:
                    if "llama-server" in arg.lower() or "llama_server" in arg.lower():
                        return proc.info["pid"]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        pass
    return None


# =============================================================================
# Inférence via l'API HTTP du serveur (compatible OpenAI)
# =============================================================================

def run_inference_via_server(
    server_url: str,
    prompt: str = BENCHMARK_PROMPT,
    max_tokens: int = None,
) -> Dict[str, Any]:
    """
    Exécute une inférence via l'API HTTP de llama-server.
    Utilise le streaming SSE pour mesurer la latence du premier token.

    L'API est compatible OpenAI : POST /v1/chat/completions

    Args:
        server_url: URL de base du serveur (ex: "http://127.0.0.1:8080").
        prompt: Texte du prompt.
        max_tokens: Nombre max de tokens à générer.

    Returns:
        Dict avec les mêmes métriques que run_single_inference().
    """
    import requests

    if max_tokens is None:
        max_tokens = INFERENCE_CONFIG["max_tokens"]

    # Mesurer la mémoire avant (processus Python + llama-server si possible)
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024**3)

    # Essayer de mesurer la mémoire du processus llama-server
    server_mem_before = 0.0
    server_pid = _find_llama_server_pid()
    if server_pid:
        try:
            server_proc = psutil.Process(server_pid)
            server_mem_before = server_proc.memory_info().rss / (1024**3)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            server_pid = None

    url = f"{server_url}/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": INFERENCE_CONFIG["temperature"],
        "top_p": INFERENCE_CONFIG["top_p"],
        "repeat_penalty": INFERENCE_CONFIG["repeat_penalty"],
        "stream": True,
        "seed": INFERENCE_CONFIG["seed"],
    }

    tokens_generated = 0
    first_token_time = None
    token_times = []
    error = None
    generated_text = ""

    try:
        start_time = time.perf_counter()

        response = requests.post(
            url,
            json=payload,
            stream=True,
            timeout=(10, 300),  # (connect_timeout, read_timeout)
        )
        response.raise_for_status()

        # Parser les Server-Sent Events (SSE)
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith("data: "):
                continue

            data_str = line[6:]  # Enlever "data: "

            if data_str.strip() == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            token_text = delta.get("content", "")

            if token_text:
                current_time = time.perf_counter()
                tokens_generated += 1
                token_times.append(current_time)

                if first_token_time is None:
                    first_token_time = current_time - start_time

                generated_text += token_text

        total_time = time.perf_counter() - start_time

    except requests.exceptions.ConnectionError:
        error = "Impossible de se connecter au serveur llama-server"
        total_time = time.perf_counter() - start_time
    except requests.exceptions.Timeout:
        error = "Timeout de la requête vers llama-server"
        total_time = time.perf_counter() - start_time
    except Exception as e:
        error = str(e)
        total_time = time.perf_counter() - start_time

    # Pas de token généré
    if tokens_generated == 0 and error is None:
        error = "Aucun token généré (le modèle a retourné un EOS immédiat)"

    mem_after = process.memory_info().rss / (1024**3)

    # Mémoire llama-server après inférence
    server_mem_after = 0.0
    if server_pid:
        try:
            server_proc = psutil.Process(server_pid)
            server_mem_after = server_proc.memory_info().rss / (1024**3)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            server_mem_after = server_mem_before

    # Mémoire totale = Python + llama-server
    total_mem_before = mem_before + server_mem_before
    total_mem_after = mem_after + server_mem_after

    result = {
        "tokens_generated": tokens_generated,
        "total_time_s": round(total_time, 4),
        "first_token_latency_s": round(first_token_time, 4) if first_token_time else None,
        "tokens_per_second": round(tokens_generated / total_time, 2) if total_time > 0 and tokens_generated > 0 else 0,
        "memory_before_gb": round(total_mem_before, 3),
        "memory_after_gb": round(total_mem_after, 3),
        "memory_delta_gb": round(total_mem_after - total_mem_before, 3),
        "server_memory_gb": round(server_mem_after, 3) if server_pid else None,
        "error": error,
        "success": error is None,
        "inference_mode": "server",
    }

    # Latence inter-tokens
    if len(token_times) > 1:
        inter_token_latencies = [
            token_times[i] - token_times[i - 1]
            for i in range(1, len(token_times))
        ]
        result["avg_inter_token_latency_ms"] = round(
            np.mean(inter_token_latencies) * 1000, 2
        )
        result["p90_inter_token_latency_ms"] = round(
            np.percentile(inter_token_latencies, 90) * 1000, 2
        )

    return result


# =============================================================================
# Benchmark complet via llama-server
# =============================================================================

def benchmark_model_server(
    model_key: str,
    server_manager: LlamaServerManager,
    backend_info: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Exécute le benchmark complet pour un modèle via llama-server.
    Le serveur est démarré automatiquement si un binary_path est configuré,
    sinon il se connecte à un serveur déjà démarré.

    Args:
        model_key: Clé du modèle dans AVAILABLE_MODELS.
        server_manager: Instance du gestionnaire de serveur.
        backend_info: Infos backend (de detect_best_backend).
        progress_callback: Callback(progress, message).

    Returns:
        Dict avec résultats au même format que benchmark_model().
    """
    from src.config import AVAILABLE_MODELS
    from src.benchmark_classic import ResourceMonitor, system_cleanup

    if model_key not in AVAILABLE_MODELS:
        return {"error": f"Modèle inconnu : {model_key}"}

    model_config = AVAILABLE_MODELS[model_key]

    # Nettoyage mémoire
    if progress_callback:
        progress_callback(0.01, f"Nettoyage mémoire avant {model_config['name']}...")
    system_cleanup(f"LLM Server {model_config['name']}")

    # Vérifier RAM
    ram_total = psutil.virtual_memory().total / (1024**3)
    if ram_total < model_config["min_ram_gb"]:
        return {
            "model": model_config["name"],
            "status": "skipped",
            "reason": f"RAM totale insuffisante ({ram_total:.1f} Go, {model_config['min_ram_gb']} Go requis)",
        }

    result = {
        "model": model_config["name"],
        "params": model_config["params"],
        "status": "running",
        "inference_mode": "server",
        "runs": [],
    }

    server_was_started = False

    try:
        # Vérifier si le serveur est déjà lancé
        if server_manager.is_running():
            if progress_callback:
                progress_callback(0.05, "Serveur llama-server déjà actif.")
        elif server_manager.binary_path:
            # Démarrer le serveur automatiquement
            from src.benchmark_ai import download_model

            if progress_callback:
                progress_callback(0.05, f"Vérification du modèle {model_config['name']}...")
            model_path = download_model(model_key)

            if progress_callback:
                progress_callback(0.08, f"Démarrage llama-server avec {model_config['name']}...")

            load_start = time.perf_counter()
            server_manager.start(
                model_path=str(model_path),
                n_gpu_layers=backend_info.get("n_gpu_layers", -1),
                n_ctx=INFERENCE_CONFIG["n_ctx"],
                progress_callback=progress_callback,
            )
            load_time = time.perf_counter() - load_start
            result["model_load_time_s"] = round(load_time, 2)
            server_was_started = True
        else:
            return {
                "model": model_config["name"],
                "status": "error",
                "error": "Serveur non démarré et aucun binaire configuré.",
            }

        result["backend"] = {
            **backend_info,
            "inference_mode": "server",
            "server_url": server_manager.base_url,
        }

        # Échauffement
        if progress_callback:
            progress_callback(0.20, f"Échauffement {model_config['name']} (serveur)...")

        for _ in range(INFERENCE_CONFIG["n_warmup_runs"]):
            run_inference_via_server(server_manager.base_url, max_tokens=32)

        # Runs de benchmark
        n_runs = INFERENCE_CONFIG["n_benchmark_runs"]
        monitor = ResourceMonitor()
        monitor.start()

        for i in range(n_runs):
            if progress_callback:
                p = 0.30 + (i / n_runs) * 0.65
                progress_callback(p, f"Inférence {model_config['name']} (serveur) — Run {i + 1}/{n_runs}")

            run_result = run_inference_via_server(server_manager.base_url)
            result["runs"].append(run_result)

        resource_usage = monitor.stop()
        result["resource_usage"] = resource_usage

        # Statistiques agrégées
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

        if progress_callback:
            progress_callback(1.0, f"Benchmark {model_config['name']} (serveur) terminé.")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        if progress_callback:
            progress_callback(1.0, f"Erreur : {e}")
    finally:
        # Arrêter le serveur si on l'a démarré
        if server_was_started:
            server_manager.stop()

    return result


def benchmark_quantization_server(
    model_key: str,
    quant_key: str,
    server_manager: LlamaServerManager,
    backend_info: Dict[str, Any],
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Exécute le benchmark pour une quantification via llama-server.

    Returns:
        Dict au même format que benchmark_single_quantization().
    """
    from src.config import AVAILABLE_MODELS
    from src.benchmark_ai import (
        get_available_quantizations,
        download_quantization,
    )
    from src.benchmark_classic import ResourceMonitor, system_cleanup

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
    system_cleanup(f"Quant Server {model_config['name']} {quant_key}")

    # Vérifier RAM
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
        "inference_mode": "server",
        "runs": [],
    }

    server_was_started = False

    try:
        # Télécharger la quantification
        if progress_callback:
            progress_callback(0.05, f"Vérification {model_config['name']} {quant_key}...")
        model_path = download_quantization(model_key, quant_key)

        actual_size_gb = model_path.stat().st_size / (1024**3)
        result["actual_file_size_gb"] = round(actual_size_gb, 3)

        # Arrêter le serveur actuel si nécessaire, puis redémarrer avec le nouveau modèle
        if server_manager.binary_path:
            if server_manager.is_running():
                server_manager.stop()
                time.sleep(1)

            if progress_callback:
                progress_callback(0.08, f"Démarrage llama-server avec {quant_key}...")

            load_start = time.perf_counter()
            server_manager.start(
                model_path=str(model_path),
                n_gpu_layers=backend_info.get("n_gpu_layers", -1),
                n_ctx=INFERENCE_CONFIG["n_ctx"],
                progress_callback=progress_callback,
            )
            load_time = time.perf_counter() - load_start
            result["model_load_time_s"] = round(load_time, 2)
            server_was_started = True
        elif server_manager.is_running():
            if progress_callback:
                progress_callback(0.10, "Serveur externe actif (modèle pré-chargé).")
        else:
            return {
                "quantization": quant_key,
                "status": "error",
                "error": "Serveur non démarré et aucun binaire configuré.",
            }

        result["backend"] = {
            **backend_info,
            "inference_mode": "server",
            "server_url": server_manager.base_url,
        }

        # Mémoire après chargement (Python + llama-server)
        process = psutil.Process()
        python_mem = process.memory_info().rss / (1024**3)
        server_pid = _find_llama_server_pid()
        server_mem = 0.0
        if server_pid:
            try:
                server_mem = psutil.Process(server_pid).memory_info().rss / (1024**3)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        result["memory_after_load_gb"] = round(python_mem + server_mem, 3)
        if server_pid:
            result["server_memory_after_load_gb"] = round(server_mem, 3)

        # Échauffement
        if progress_callback:
            progress_callback(0.20, f"Échauffement {quant_key} (serveur)...")
        for _ in range(INFERENCE_CONFIG["n_warmup_runs"]):
            run_inference_via_server(server_manager.base_url, max_tokens=32)

        # Runs de benchmark
        n_runs = INFERENCE_CONFIG["n_benchmark_runs"]
        monitor = ResourceMonitor()
        monitor.start()

        for i in range(n_runs):
            if progress_callback:
                p = 0.30 + (i / n_runs) * 0.65
                progress_callback(p, f"{quant_key} (serveur) — Run {i + 1}/{n_runs}")

            run_result = run_inference_via_server(server_manager.base_url)
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

        if progress_callback:
            progress_callback(1.0, f"{quant_key} (serveur) terminé.")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        if progress_callback:
            progress_callback(1.0, f"Erreur {quant_key}: {e}")
    finally:
        if server_was_started:
            server_manager.stop()

    return result


def run_all_server_benchmarks(
    model_keys: List[str] = None,
    quantization_models: Dict[str, List[str]] = None,
    server_manager: LlamaServerManager = None,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Exécute tous les benchmarks IA via llama-server.
    Point d'entrée principal pour le mode serveur.

    Args:
        model_keys: Liste des modèles à tester.
        quantization_models: Dict {model_key: [quant_key, ...]}.
        server_manager: Instance LlamaServerManager configurée.
        progress_callback: Callback(progress, message).

    Returns:
        Dict au même format que run_all_ai_benchmarks().
    """
    from src.config import AVAILABLE_MODELS
    from src.benchmark_ai import (
        detect_best_backend,
        get_compatible_models,
    )
    from src.benchmark_classic import system_cleanup

    if server_manager is None:
        binary = find_llama_server_binary()
        server_manager = LlamaServerManager(binary_path=binary)

    # Déterminer les modèles
    if model_keys is None:
        ram_total = psutil.virtual_memory().total / (1024**3)
        compatible = get_compatible_models(ram_total)
        model_keys = list(compatible.keys())

    if not model_keys and not quantization_models:
        return {
            "type": "ai_benchmarks",
            "status": "skipped",
            "reason": "Aucun modèle compatible.",
        }

    backend_info = detect_best_backend()
    backend_info["inference_mode"] = "server"

    n_model_tasks = len(model_keys) if model_keys else 0
    n_quant_tasks = sum(len(qs) for qs in (quantization_models or {}).values())
    total_tasks = n_model_tasks + n_quant_tasks
    if total_tasks == 0:
        total_tasks = 1

    all_results = {}
    start_time = time.time()

    # Benchmarks par modèle
    for idx, model_key in enumerate(model_keys or []):
        def model_progress(p, msg, _idx=idx):
            if progress_callback:
                overall = (_idx + p) / total_tasks
                progress_callback(overall, msg)

        if progress_callback:
            progress_callback(
                idx / total_tasks,
                f"Modèle {idx + 1}/{n_model_tasks}: "
                f"{AVAILABLE_MODELS.get(model_key, {}).get('name', model_key)} (serveur)",
            )

        if idx > 0:
            system_cleanup(f"entre modèles serveur ({idx + 1}/{n_model_tasks})")

        all_results[model_key] = benchmark_model_server(
            model_key, server_manager, backend_info,
            progress_callback=model_progress,
        )

    # Comparaison de quantification
    quant_comparison = {}
    if quantization_models:
        task_offset = n_model_tasks

        for model_key, quant_keys in quantization_models.items():
            n_q = len(quant_keys)
            model_name = AVAILABLE_MODELS.get(model_key, {}).get("name", model_key)

            comparison = {
                "model_name": model_name,
                "model_key": model_key,
                "params": AVAILABLE_MODELS.get(model_key, {}).get("params", "?"),
                "variants_tested": quant_keys,
                "results": {},
            }

            for qi, quant_key in enumerate(quant_keys):
                def quant_progress(p, msg, _offset=task_offset, _qi=qi):
                    if progress_callback:
                        overall = (_offset + _qi + p) / total_tasks
                        progress_callback(min(overall, 1.0), msg)

                if progress_callback:
                    progress_callback(
                        (task_offset + qi) / total_tasks,
                        f"Quantification {qi + 1}/{n_q} : {model_name} {quant_key} (serveur)",
                    )

                if qi > 0:
                    system_cleanup(f"entre quantifications serveur ({quant_key})")

                comparison["results"][quant_key] = benchmark_quantization_server(
                    model_key, quant_key, server_manager, backend_info,
                    progress_callback=quant_progress,
                )

            # Tableau comparatif
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

            quant_comparison[model_key] = comparison
            task_offset += n_q

    elapsed_total = time.time() - start_time

    result = {
        "type": "ai_benchmarks",
        "inference_mode": "server",
        "prompt": BENCHMARK_PROMPT,
        "inference_config": INFERENCE_CONFIG,
        "total_time_s": round(elapsed_total, 2),
        "models_tested": len(model_keys or []),
        "results": all_results,
    }

    if quant_comparison:
        result["quantization_comparison"] = quant_comparison

    return result


# =============================================================================
# Utilitaires
# =============================================================================

def check_server_status(
    host: str = None,
    port: int = None,
) -> Dict[str, Any]:
    """
    Vérifie le statut d'un serveur llama-server.

    Returns:
        Dict avec status, url, et infos serveur si disponible.
    """
    host = host or LLAMA_SERVER_CONFIG["host"]
    port = port or LLAMA_SERVER_CONFIG["port"]
    url = f"http://{host}:{port}"

    status = {
        "url": url,
        "running": False,
        "info": None,
    }

    try:
        import requests
        resp = requests.get(f"{url}/health", timeout=3)
        if resp.status_code == 200:
            status["running"] = True

            # Essayer de récupérer les infos du serveur
            try:
                props = requests.get(f"{url}/props", timeout=3)
                if props.status_code == 200:
                    status["info"] = props.json()
            except Exception:
                pass
    except Exception:
        pass

    return status


def get_llama_cpp_releases_url() -> str:
    """Retourne l'URL des releases GitHub de llama.cpp."""
    return "https://github.com/ggerganov/llama.cpp/releases"
