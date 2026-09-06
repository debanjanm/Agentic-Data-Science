"""Runnable self-check for the AutoML core (no framework — plain asserts).

    python tests/test_modeling.py
"""

import pandas as pd
from sklearn.datasets import load_diabetes, load_iris

from agentic_ds.modeling import CLASSIFICATION_ZOO, detect_task_type, feature_importance, predict, shap_importance, train_leaderboard, tune_model


def test_classification():
    df = load_iris(as_frame=True).frame
    assert detect_task_type(df, "target") == "classification"

    result = train_leaderboard(df, "target", "classification", model_names=["logistic_regression", "random_forest"])
    leaderboard = result["leaderboard"]
    assert len(leaderboard) == 2
    assert "accuracy" in leaderboard.columns
    assert leaderboard.iloc[0]["accuracy"] >= leaderboard.iloc[-1]["accuracy"]

    best_name = leaderboard.iloc[0]["model"]
    best_pipeline = result["pipelines"][best_name]
    importances = feature_importance(best_pipeline, result["X_test"], result["y_test"], "classification")
    assert len(importances) == 4  # iris has 4 features

    preds = predict(best_pipeline, result["X_test"], "classification")
    assert "prediction" in preds.columns and "confidence" in preds.columns


def test_regression_and_tuning():
    df = load_diabetes(as_frame=True).frame
    assert detect_task_type(df, "target") == "regression"

    result = train_leaderboard(df, "target", "regression", model_names=["linear_regression", "knn"])
    assert "r2" in result["leaderboard"].columns

    tuned = tune_model("knn", result["pipelines"]["knn"], result["X_train"], result["y_train"], "regression", n_iter=3)
    preds = predict(tuned, result["X_test"], "regression")
    assert "prediction" in preds.columns and "confidence" not in preds.columns


def test_xgboost_and_lightgbm_with_string_labels():
    # Regression check for the XGBoost label-encoding wrapper: XGBClassifier
    # errors on non-0..n-1 labels, unlike every other classifier in the zoo.
    if "xgboost" not in CLASSIFICATION_ZOO or "lightgbm" not in CLASSIFICATION_ZOO:
        return  # optional deps not installed — nothing to check
    df = pd.DataFrame({
        "a": list(range(20)),
        "b": list(range(20, 0, -1)),
        "label": (["yes", "no"] * 10),
    })
    result = train_leaderboard(df, "label", "classification", model_names=["xgboost", "lightgbm"])
    assert set(result["leaderboard"]["model"]) == {"xgboost", "lightgbm"}
    preds = result["pipelines"]["xgboost"].predict(result["X_test"])
    assert set(preds) <= {"yes", "no"}


def test_shap_importance():
    if "xgboost" not in CLASSIFICATION_ZOO:
        return  # shap ships alongside xgboost/lightgbm in requirements.txt
    df = load_iris(as_frame=True).frame
    result = train_leaderboard(df, "target", "classification", model_names=["random_forest"])
    importances = shap_importance(result["pipelines"]["random_forest"], result["X_test"], "classification")
    assert len(importances) == 4
    assert (importances["mean_abs_shap"] >= 0).all()


if __name__ == "__main__":
    test_classification()
    test_regression_and_tuning()
    test_xgboost_and_lightgbm_with_string_labels()
    test_shap_importance()
    print("OK — all modeling self-checks passed.")
