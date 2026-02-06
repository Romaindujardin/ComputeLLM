"""
ComputeLLM - Point d'entrée CLI.
Permet de lancer l'interface Streamlit ou d'exécuter les benchmarks en ligne de commande.
"""

import argparse
import json
import sys
import os
from pathlib import Path

# S'assurer que le répertoire racine est dans le path
sys.path.insert(0, str(Path(__file__).parent))


def launch_gui():
    """Lance l'interface Streamlit."""
    os.system(f"{sys.executable} -m streamlit run app.py --server.headless true")


def run_cli_benchmarks(models=None, skip_classic=False, skip_ai=False, output=None):
    """Exécute les benchmarks en mode CLI."""
    from src.hardware_detect import get_full_hardware_info, get_hardware_summary
    from src.benchmark_classic import run_all_classic_benchmarks
    from src.benchmark_ai import run_all_ai_benchmarks, get_compatible_models
    from src.results_manager import save_results

    def progress(p, msg):
        bar = "█" * int(p * 40) + "░" * (40 - int(p * 40))
        print(f"\r  [{bar}] {p*100:5.1f}% | {msg:<60}", end="", flush=True)

    # 1. Détection matérielle
    print("=" * 70)
    print("  ComputeLLM - AI Hardware Benchmark")
    print("=" * 70)
    print("\n[1/3] Détection matérielle...")
    hw_info = get_full_hardware_info()
    print(get_hardware_summary(hw_info))

    # 2. Benchmarks classiques
    classic_results = None
    if not skip_classic:
        print("\n[2/3] Benchmarks classiques...")
        classic_results = run_all_classic_benchmarks(progress_callback=progress)
        print(f"\n  ✅ Terminé en {classic_results['total_time_s']:.1f}s\n")
    else:
        print("\n[2/3] Benchmarks classiques : IGNORÉS")

    # 3. Benchmarks IA
    ai_results = None
    if not skip_ai:
        print("\n[3/3] Benchmarks IA (inférence LLM)...")

        if models is None:
            import psutil
            ram_total = psutil.virtual_memory().total / (1024**3)
            compatible = get_compatible_models(ram_total)
            # Par défaut, ne tester que le plus petit
            models = [list(compatible.keys())[0]] if compatible else []

        if models:
            ai_results = run_all_ai_benchmarks(
                model_keys=models,
                progress_callback=progress,
            )
            print(f"\n  ✅ Terminé en {ai_results['total_time_s']:.1f}s\n")
        else:
            print("  ⚠️ Aucun modèle compatible trouvé.")
    else:
        print("\n[3/3] Benchmarks IA : IGNORÉS")

    # 4. Sauvegarde
    print("\n[Sauvegarde]")
    save_path = save_results(
        hardware_info=hw_info,
        classic_results=classic_results,
        ai_results=ai_results,
    )
    print(f"  ✅ Résultats sauvegardés : {save_path}")

    if output:
        import shutil
        shutil.copy(save_path, output)
        print(f"  ✅ Copie vers : {output}")

    print("\n" + "=" * 70)
    print("  Benchmark terminé ! Lancez l'interface pour visualiser les résultats :")
    print("  python main.py --gui")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="ComputeLLM - AI Hardware Benchmark Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python main.py --gui                          Lancer l'interface graphique
  python main.py --cli                          Exécuter tous les benchmarks (CLI)
  python main.py --cli --models tinyllama-1.1b  Tester un modèle spécifique
  python main.py --cli --skip-ai                Benchmarks classiques uniquement
  python main.py --cli --skip-classic           Benchmarks IA uniquement
        """,
    )

    parser.add_argument(
        "--gui", action="store_true",
        help="Lancer l'interface graphique Streamlit",
    )
    parser.add_argument(
        "--cli", action="store_true",
        help="Exécuter les benchmarks en ligne de commande",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Modèles à tester (ex: tinyllama-1.1b mistral-7b)",
    )
    parser.add_argument(
        "--skip-classic", action="store_true",
        help="Ignorer les benchmarks classiques",
    )
    parser.add_argument(
        "--skip-ai", action="store_true",
        help="Ignorer les benchmarks IA",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Chemin de sortie pour le fichier de résultats",
    )
    parser.add_argument(
        "--detect", action="store_true",
        help="Afficher uniquement les informations matérielles",
    )

    args = parser.parse_args()

    if args.detect:
        from src.hardware_detect import get_full_hardware_info, get_hardware_summary
        hw = get_full_hardware_info()
        print(get_hardware_summary(hw))
        return

    if args.cli:
        run_cli_benchmarks(
            models=args.models,
            skip_classic=args.skip_classic,
            skip_ai=args.skip_ai,
            output=args.output,
        )
    elif args.gui or len(sys.argv) == 1:
        launch_gui()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
