"""Reproduit toute l'analyse en une commande : python scripts/run_all.py

Chaque etape tourne dans son propre processus Python : toute sa memoire est
rendue au systeme a la fin de l'etape, avant la suivante. Si une etape
echoue, la chaine s'arrete et indique laquelle.

    python scripts/run_all.py                         # toutes les etapes
    python scripts/run_all.py --steps robustness business
    python scripts/run_all.py --list                  # noms des etapes

L'EDA BigQuery (src.bq_eda) est a part car elle demande un acces GCP.
Toutes les etapes utilisent random_state=42 (seeds fixes).
"""
from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# nom court -> commande (module + arguments)
STEPS: dict[str, list[str]] = {
    "train": ["src.train"],
    "evaluate": ["src.evaluate"],
    "interpret": ["src.interpret"],
    "segmentation": ["src.segmentation"],
    "interactions": ["src.interactions"],
    "robustness": ["src.robustness"],
    "robustness-segments": ["src.robustness", "--segments-only"],
    "business": ["src.business"],
    "mlflow": ["src.log_analysis"],
}
DEFAULT_ORDER = ["train", "evaluate", "interpret", "segmentation", "interactions",
                 "robustness", "business", "mlflow"]


def run_steps(steps: list[str], runner=subprocess.run) -> int:
    """Lance les etapes dans l'ordre ; s'arrete a la premiere qui echoue.
    Retourne 0 si tout passe, sinon le code de sortie de l'etape en echec."""
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        print(f"Etape(s) inconnue(s) : {unknown}. Disponibles : {list(STEPS)}")
        return 2
    env = os.environ | {"PYTHONPATH": str(ROOT), "MLFLOW_DISABLE_AGENT_HINT": "1"}
    for step in steps:
        t0 = time.time()
        print(f"\n===== {step} : python -m {' '.join(STEPS[step])}", flush=True)
        result = runner([sys.executable, "-m", *STEPS[step]], cwd=ROOT, env=env)
        gc.collect()
        if result.returncode != 0:
            print(f"\n===== ECHEC a l'etape '{step}' (code {result.returncode}). "
                  f"Etapes non lancees : {steps[steps.index(step) + 1:]}", flush=True)
            return result.returncode
        print(f"===== {step} : OK en {time.time() - t0:.0f} s", flush=True)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", nargs="+", default=DEFAULT_ORDER,
                        help="etapes a lancer, dans l'ordre donne (defaut : toutes)")
    parser.add_argument("--list", action="store_true", help="affiche les etapes disponibles")
    args = parser.parse_args()
    if args.list:
        for name, cmd in STEPS.items():
            print(f"{name:20s} python -m {' '.join(cmd)}")
        return
    sys.exit(run_steps(args.steps))


if __name__ == "__main__":
    main()
