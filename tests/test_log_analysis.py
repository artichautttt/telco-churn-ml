import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.log_analysis import collect_metrics


def test_collect_metrics_empty_dir(tmp_path):
    assert collect_metrics(tmp_path) == {}


def test_collect_metrics_reads_business_summary(tmp_path):
    (tmp_path / "business").mkdir()
    (tmp_path / "business" / "summary.json").write_text(json.dumps({
        "optimal_threshold_train_oof": 0.3,
        "at_optimal_threshold": {"total_cost_test": 100.0, "recall": 0.9},
        "at_default_threshold_0_5": {"total_cost_test": 120.0},
        "capture_top10": 0.3, "capture_top20": 0.5, "lift_top10": 3.0, "lift_top20": 2.5,
    }))
    m = collect_metrics(tmp_path)
    assert m["cost_optimal_threshold"] == 0.3
    assert m["lift_top20"] == 2.5
    assert all(isinstance(v, float) for v in m.values())
