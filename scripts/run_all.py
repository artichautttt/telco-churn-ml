"""Reproduit toute l'analyse en une commande : python scripts/run_all.py

Ordre : entrainement/selection -> evaluation -> interpretabilite ->
segmentation -> interactions -> robustesse -> lecture metier.
L'EDA BigQuery (src.bq_eda) est a part car elle demande un acces GCP :
    python -m src.bq_eda --project MON_PROJET
Toutes les etapes utilisent random_state=42 (seeds fixes).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    "src.train",
    "src.evaluate",
    "src.interpret",
    "src.segmentation",
    "src.interactions",
    "src.robustness",
    "src.business",
]


def main() -> None:
    env = os.environ | {"PYTHONPATH": str(ROOT), "MLFLOW_DISABLE_AGENT_HINT": "1"}
    steps = sys.argv[1:] or STEPS
    for step in steps:
        t0 = time.time()
        print(f"\n===== python -m {step}", flush=True)
        subprocess.run([sys.executable, "-m", step], cwd=ROOT, env=env, check=True)
        print(f"===== {step} : {time.time() - t0:.0f} s", flush=True)
    if steps == STEPS:
        subprocess.run([sys.executable, "-m", "src.log_analysis"], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
