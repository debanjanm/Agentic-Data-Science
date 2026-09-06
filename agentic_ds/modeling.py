"""The AutoML core: task detection, baseline model zoo, training,
leaderboard, tuning, feature importance, and prediction.

Plain scikit-learn — no PyCaret/AutoGluon (see PLAN.md "Modeling stack"
for why). Every function takes/returns plain pandas/sklearn objects so it
works standalone, with or without the LLM narrative layer.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR

CLASSIFICATION_ZOO = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
    "knn": KNeighborsClassifier(),
    "svm": SVC(probability=True, random_state=42),
}

REGRESSION_ZOO = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(random_state=42),
    "gradient_boosting": GradientBoostingRegressor(random_state=42),
    "knn": KNeighborsRegressor(),
    "svm": SVR(),
}

PARAM_GRIDS = {
    "logistic_regression": {"model__C": [0.01, 0.1, 1, 10, 100]},
    "random_forest": {"model__n_estimators": [100, 200, 400], "model__max_depth": [None, 5, 10, 20]},
    "gradient_boosting": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.01, 0.1, 0.2],
        "model__max_depth": [2, 3, 4],
    },
    "knn": {"model__n_neighbors": [3, 5, 7, 9, 11]},
    "svm": {"model__C": [0.1, 1, 10], "model__gamma": ["scale", "auto"]},
}

_GBM_PARAM_GRID = {
    "model__n_estimators": [100, 200, 400],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.01, 0.1, 0.2],
}


class _XGBClassifierSafe(BaseEstimator, ClassifierMixin):
    """XGBClassifier (unlike every other classifier in the zoo) requires
    0..n-1 integer labels and errors on arbitrary string/int class labels.
    This wraps it so it accepts the same labels everything else does —
    hyperparameters are declared explicitly (not **kwargs) so sklearn's
    clone/get_params/set_params still work for RandomizedSearchCV tuning.
    """

    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y):
        from xgboost import XGBClassifier

        self._encoder = LabelEncoder().fit(y)
        self._model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            eval_metric="logloss",
        )
        self._model.fit(X, self._encoder.transform(y))
        self.classes_ = self._encoder.classes_
        return self

    def predict(self, X):
        return self._encoder.inverse_transform(self._model.predict(X))

    def predict_proba(self, X):
        return self._model.predict_proba(X)


try:
    from xgboost import XGBRegressor

    CLASSIFICATION_ZOO["xgboost"] = _XGBClassifierSafe()
    REGRESSION_ZOO["xgboost"] = XGBRegressor(random_state=42)
    PARAM_GRIDS["xgboost"] = dict(_GBM_PARAM_GRID)
except ImportError:
    pass

try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    CLASSIFICATION_ZOO["lightgbm"] = LGBMClassifier(random_state=42, verbose=-1)
    REGRESSION_ZOO["lightgbm"] = LGBMRegressor(random_state=42, verbose=-1)
    PARAM_GRIDS["lightgbm"] = dict(_GBM_PARAM_GRID)
except ImportError:
    pass


def detect_task_type(df: pd.DataFrame, target: str) -> str:
    """Heuristic: object/category/bool or low-cardinality numeric -> classification."""
    series = df[target]
    if series.dtype == "object" or series.dtype.name in ("category", "bool"):
        return "classification"
    n_unique = series.nunique()
    if n_unique <= 20 and n_unique / len(series) < 0.05:
        return "classification"
    return "regression"


def _model_zoo(task_type: str) -> dict:
    return CLASSIFICATION_ZOO if task_type == "classification" else REGRESSION_ZOO


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])


def _score(task_type: str, y_true, y_pred, y_proba=None) -> dict:
    if task_type == "classification":
        metrics = {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "f1_weighted": round(f1_score(y_true, y_pred, average="weighted"), 4),
        }
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics["roc_auc"] = round(roc_auc_score(y_true, y_proba[:, 1]), 4)
            except (ValueError, IndexError):
                pass
        return metrics
    return {
        "r2": round(r2_score(y_true, y_pred), 4),
        "rmse": round(mean_squared_error(y_true, y_pred) ** 0.5, 4),
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
    }


def train_leaderboard(
    df: pd.DataFrame,
    target: str,
    task_type: str,
    model_names: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train every model in `model_names` (default: the whole zoo for the
    task type) and return a leaderboard + everything needed to inspect or
    tune each one.
    """
    X = df.drop(columns=[target])
    y = df[target]

    stratify = y if task_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    zoo = _model_zoo(task_type)
    model_names = model_names or list(zoo)

    rows = []
    pipelines = {}
    for name in model_names:
        pipeline = Pipeline([("preprocess", build_preprocessor(X_train)), ("model", zoo[name])])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
        metrics = _score(task_type, y_test, y_pred, y_proba)

        rows.append({"model": name, **metrics})
        pipelines[name] = pipeline

    primary_metric = "accuracy" if task_type == "classification" else "r2"
    leaderboard = pd.DataFrame(rows).sort_values(primary_metric, ascending=False).reset_index(drop=True)

    return {
        "leaderboard": leaderboard,
        "pipelines": pipelines,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "primary_metric": primary_metric,
        "task_type": task_type,
        "target": target,
    }


def score_pipeline(pipeline: Pipeline, X_test: pd.DataFrame, y_test, task_type: str) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
    return _score(task_type, y_test, y_pred, y_proba)


def tune_model(name: str, pipeline: Pipeline, X_train, y_train, task_type: str, n_iter: int = 10) -> Pipeline:
    grid = PARAM_GRIDS.get(name)
    if not grid:
        return pipeline  # nothing to tune (e.g. linear_regression)
    scoring = "accuracy" if task_type == "classification" else "r2"
    search = RandomizedSearchCV(
        pipeline, grid, n_iter=n_iter, scoring=scoring, cv=3, random_state=42, n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def feature_importance(pipeline: Pipeline, X_test: pd.DataFrame, y_test, task_type: str) -> pd.DataFrame:
    scoring = "accuracy" if task_type == "classification" else "r2"
    result = permutation_importance(pipeline, X_test, y_test, n_repeats=10, random_state=42, scoring=scoring)
    return (
        pd.DataFrame({"feature": X_test.columns, "importance": result.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def shap_importance(
    pipeline: Pipeline, X_test: pd.DataFrame, task_type: str, sample_size: int = 30, random_state: int = 42
) -> pd.DataFrame:
    """Mean |SHAP value| per feature — a second, different-methodology cross-check
    on top of permutation_importance. Model-agnostic (works through the whole
    pipeline, any estimator), so it's computed on a small sample: it calls the
    pipeline many times per row and gets slow on wide data at full size.

    For classification this explains "what drives confidence in whatever class
    was predicted" (max class probability) rather than one specific class —
    simpler than juggling per-class SHAP values for the multiclass case.
    """
    import shap  # heavy import, optional dependency — lazy per Shonku's generate_chart pattern

    sample = X_test.sample(min(sample_size, len(X_test)), random_state=random_state)
    if task_type == "classification" and hasattr(pipeline, "predict_proba"):
        score_fn = lambda x: pipeline.predict_proba(x).max(axis=1)  # noqa: E731
    else:
        score_fn = pipeline.predict

    explainer = shap.Explainer(score_fn, sample)
    shap_values = explainer(sample)
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    return (
        pd.DataFrame({"feature": sample.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def predict(pipeline: Pipeline, new_df: pd.DataFrame, task_type: str) -> pd.DataFrame:
    result = new_df.copy()
    result["prediction"] = pipeline.predict(new_df)
    if task_type == "classification" and hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(new_df)
        result["confidence"] = proba.max(axis=1).round(4)
    return result
