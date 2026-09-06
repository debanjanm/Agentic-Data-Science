"""Data cleaning: propose a plan, let the user approve/edit it, apply it.

Deliberately rule-based, not LLM-driven — these are mechanical, deterministic
decisions (impute median, drop a near-duplicate-of-ID column, etc.) that
don't benefit from an LLM call, matching the "propose then apply, never
silent mutation" rule from PLAN.md. Encoding is NOT handled here — that
stays inside modeling.build_preprocessor's pipeline, so it's never silently
out of sync with what actually gets fit.
"""

import pandas as pd

from agentic_ds.eda import detect_outliers

HIGH_MISSING_PCT = 50.0
HIGH_CARDINALITY_RATIO = 0.5
HIGH_CARDINALITY_MIN_UNIQUE = 50
OUTLIER_PCT_THRESHOLD = 10.0


def propose_plan(df: pd.DataFrame, target: str) -> list[dict]:
    """Return a list of proposed cleaning steps, each independently approvable."""
    steps: list[dict] = []

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        steps.append({
            "column": "(all rows)",
            "issue": f"{duplicate_count} duplicate row(s)",
            "action": "drop_duplicates",
            "approved": True,
        })

    for column in df.columns:
        if column == target:
            continue
        series = df[column]
        is_numeric = pd.api.types.is_numeric_dtype(series)

        missing_pct = round(series.isna().mean() * 100, 1)
        if missing_pct > 0:
            if missing_pct > HIGH_MISSING_PCT:
                steps.append({
                    "column": column,
                    "issue": f"{missing_pct}% missing",
                    "action": "drop_column",
                    "approved": True,
                })
                continue  # no point proposing further steps for a dropped column
            action = "impute_median" if is_numeric else "impute_mode"
            steps.append({
                "column": column,
                "issue": f"{missing_pct}% missing",
                "action": action,
                "approved": True,
            })

        if not is_numeric:
            n_unique = series.nunique()
            if n_unique >= HIGH_CARDINALITY_MIN_UNIQUE and n_unique / len(series) > HIGH_CARDINALITY_RATIO:
                steps.append({
                    "column": column,
                    "issue": f"{n_unique} unique values (likely an ID, not a feature)",
                    "action": "drop_column",
                    "approved": True,
                })

        if is_numeric:
            outliers = detect_outliers(df, column)
            if outliers.get("outlier_pct", 0) > OUTLIER_PCT_THRESHOLD:
                steps.append({
                    "column": column,
                    "issue": f"{outliers['outlier_pct']}% outliers (IQR method)",
                    "action": "cap_outliers",
                    "approved": True,
                })

    return steps


def apply_plan(df: pd.DataFrame, steps: list[dict]) -> pd.DataFrame:
    """Apply only the steps with approved=True, in a safe order (drops first)."""
    result = df.copy()

    approved = [s for s in steps if s["approved"]]

    if any(s["action"] == "drop_duplicates" for s in approved):
        result = result.drop_duplicates().reset_index(drop=True)

    drop_columns = [s["column"] for s in approved if s["action"] == "drop_column"]
    if drop_columns:
        result = result.drop(columns=[c for c in drop_columns if c in result.columns])

    for step in approved:
        column = step["column"]
        if column not in result.columns:
            continue  # already dropped above
        if step["action"] == "impute_median":
            result[column] = result[column].fillna(result[column].median())
        elif step["action"] == "impute_mode":
            mode = result[column].mode(dropna=True)
            if not mode.empty:
                result[column] = result[column].fillna(mode.iloc[0])
        elif step["action"] == "cap_outliers":
            outliers = detect_outliers(result, column)
            result[column] = result[column].clip(lower=outliers["lower_fence"], upper=outliers["upper_fence"])

    return result
