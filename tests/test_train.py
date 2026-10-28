import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import compute_scale_pos_weight, make_models, select_model


def cv(auc_recall: dict) -> dict:
    return {
        name: {"roc_auc": {"mean": auc, "std": 0.0}, "recall": {"mean": rec, "std": 0.0}}
        for name, (auc, rec) in auc_recall.items()
    }


def test_scale_pos_weight_is_negative_over_positive_ratio():
    assert compute_scale_pos_weight(np.array([0, 0, 0, 1])) == pytest.approx(3.0)


def test_scale_pos_weight_without_positive_raises():
    with pytest.raises(ValueError):
        compute_scale_pos_weight([0, 0])


def test_all_models_are_rebalanced():
    models = make_models(2.5)
    assert models["logistic_regression"].class_weight == "balanced"
    assert models["random_forest"].class_weight == "balanced"
    assert models["xgboost"].get_params()["scale_pos_weight"] == 2.5


def test_reference_kept_when_auc_gap_small_and_recall_equivalent():
    res = cv({"logistic_regression": (0.840, 0.80), "xgboost": (0.845, 0.79)})
    assert select_model(res)[0] == "logistic_regression"


def test_best_model_chosen_when_auc_gap_too_large():
    res = cv({"logistic_regression": (0.820, 0.80), "xgboost": (0.835, 0.79)})
    assert select_model(res)[0] == "xgboost"


def test_best_model_chosen_when_reference_recall_clearly_lower():
    res = cv({"logistic_regression": (0.840, 0.70), "xgboost": (0.845, 0.80)})
    assert select_model(res)[0] == "xgboost"


def test_reference_kept_when_it_is_the_best():
    res = cv({"logistic_regression": (0.85, 0.80), "random_forest": (0.82, 0.65)})
    name, why = select_model(res)
    assert name == "logistic_regression"
    assert "conserve" in why
