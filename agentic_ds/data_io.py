"""Dataset loading: built-in sklearn datasets or an uploaded CSV.

Kept dependency-light on purpose (sklearn only, no seaborn) — enough to
demo both classification and regression task types.
"""

import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer, load_diabetes, load_iris, load_wine

BUILT_IN_DATASETS = {
    "iris": "Iris — flower species classification (150 rows, 5 cols)",
    "wine": "Wine — wine quality recognition (178 rows, 14 cols)",
    "breast_cancer": "Breast cancer — tumor classification (569 rows, 31 cols)",
    "diabetes": "Diabetes — disease progression regression (442 rows, 11 cols)",
    "titanic": "Titanic — passenger survival (1309 rows, 14 cols, needs internet)",
}

# Suggested target column per built-in — the loader always puts the target
# as the last column, but the UI needs a name to preselect.
BUILT_IN_TARGETS = {
    "iris": "target",
    "wine": "target",
    "breast_cancer": "target",
    "diabetes": "target",
    "titanic": "survived",
}


def load_builtin(name: str) -> pd.DataFrame:
    name = name.lower()
    if name == "iris":
        return load_iris(as_frame=True).frame.copy()
    if name == "wine":
        return load_wine(as_frame=True).frame.copy()
    if name == "breast_cancer":
        return load_breast_cancer(as_frame=True).frame.copy()
    if name == "diabetes":
        return load_diabetes(as_frame=True).frame.copy()
    if name == "titanic":
        return fetch_openml(name="titanic", version=1, as_frame=True).frame.copy()
    raise ValueError(f"Unknown built-in dataset '{name}'. Choose one of: {list(BUILT_IN_DATASETS)}")


def load_csv(file) -> pd.DataFrame:
    """`file` is a path or a Streamlit UploadedFile (both support pandas.read_csv)."""
    return pd.read_csv(file)
