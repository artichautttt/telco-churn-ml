import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.build_serving import build, calibration_method


def test_calibration_method_default_and_from_summary(tmp_path):
    assert calibration_method(tmp_path / "absent.json") == "sigmoid"
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"calibration": {"chosen_method": "isotonic"}}))
    assert calibration_method(summary) == "isotonic"


def test_build_writes_all_serving_artifacts(tmp_path):
    # Sous-echantillon stratifie du jeu reel pour garder le test rapide.
    raw = pd.read_csv(ROOT / "data" / "raw" / "telco_churn.csv")
    small = raw.groupby("Churn", group_keys=False).sample(frac=0.15, random_state=0)
    data = tmp_path / "small.csv"
    small.to_csv(data, index=False)

    out = build(str(data), tmp_path / "models", "sigmoid")
    for name in ("best_model.pkl", "calibrated_model.pkl", "test_split.pkl", "serving_config.json"):
        assert (tmp_path / "models" / name).exists(), name
    config = json.loads(out["config"].read_text())
    assert 0 < config["threshold"] < 1
    assert config["model_path"].endswith("calibrated_model.pkl")
