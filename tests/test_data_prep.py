"""Runnable self-check for the data-prep propose/apply logic.

    python -m tests.test_data_prep
"""

import numpy as np
import pandas as pd

from agentic_ds.data_prep import apply_plan, propose_plan


def test_propose_and_apply():
    df = pd.DataFrame({
        "id": [f"id_{i}" for i in range(100)],  # high-cardinality -> drop
        "age": [25, 30, np.nan] * 33 + [40],  # missing -> impute
        "city": ["NYC", "LA", None] * 33 + ["NYC"],  # missing categorical -> impute
        "income": list(range(99)) + [100_000],  # one big outlier
        "target": [0, 1] * 50,
    })
    df = pd.concat([df, df.iloc[[0]]])  # one duplicate row

    steps = propose_plan(df, target="target")
    actions = {s["action"] for s in steps}
    assert "drop_duplicates" in actions
    assert "impute_median" in actions  # age
    assert "impute_mode" in actions  # city
    assert any(s["action"] == "drop_column" and s["column"] == "id" for s in steps)

    cleaned = apply_plan(df, steps)
    assert "id" not in cleaned.columns
    assert cleaned["age"].isna().sum() == 0
    assert cleaned["city"].isna().sum() == 0
    assert len(cleaned) < len(df)  # duplicate dropped
    assert "target" in cleaned.columns  # never touched


def test_clean_data_proposes_nothing():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    assert propose_plan(df, target="target") == []


if __name__ == "__main__":
    test_propose_and_apply()
    test_clean_data_proposes_nothing()
    print("OK — all data-prep self-checks passed.")
