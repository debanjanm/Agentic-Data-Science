"""EDA / profiling functions. Ported in spirit from Project-Shonku's
data_analyst tools.py, but returning plain pandas objects instead of JSON
strings — this app renders them directly with native Streamlit widgets
instead of calling them as LLM tools (no agent needed for basic profiling).
"""

import pandas as pd


def schema_overview(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    return pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing": missing.values,
        "missing_pct": (missing / len(df) * 100).round(2).values,
        "n_unique": [df[c].nunique() for c in df.columns],
    })


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.describe().T.round(3)


def top_value_counts(df: pd.DataFrame, column: str, limit: int = 10) -> pd.Series:
    return df[column].astype(str).value_counts(dropna=False).head(limit)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()
    return numeric_df.corr().round(3)


def top_correlated_pairs(corr: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if corr.empty:
        return pd.DataFrame(columns=["column_a", "column_b", "correlation"])
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], float(corr.iloc[i, j])))
    pairs.sort(key=lambda p: abs(p[2]), reverse=True)
    return pd.DataFrame(pairs[:n], columns=["column_a", "column_b", "correlation"])


def detect_outliers(df: pd.DataFrame, column: str) -> dict:
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return {"column": column, "error": "no numeric values"}
    q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    return {
        "column": column,
        "total_rows": int(len(series)),
        "outlier_count": int(len(outliers)),
        "outlier_pct": round(len(outliers) / len(series) * 100, 2),
        "lower_fence": round(lower, 4),
        "upper_fence": round(upper, 4),
    }


def rule_based_narrative(df: pd.DataFrame) -> list[str]:
    """No-LLM fallback summary — always available, no API key needed."""
    bullets = [f"{len(df):,} rows, {len(df.columns)} columns."]

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        worst = missing.index[0]
        pct = round(missing.iloc[0] / len(df) * 100, 1)
        bullets.append(f"{len(missing)} column(s) have missing values — worst is '{worst}' at {pct}%.")
    else:
        bullets.append("No missing values.")

    corr = correlation_matrix(df)
    pairs = top_correlated_pairs(corr, n=1)
    if not pairs.empty:
        row = pairs.iloc[0]
        bullets.append(
            f"Strongest correlation: '{row.column_a}' vs '{row.column_b}' = {row.correlation:.2f}."
        )

    numeric_cols = df.select_dtypes(include="number").columns
    flagged = []
    for col in numeric_cols[:20]:
        result = detect_outliers(df, col)
        if result.get("outlier_pct", 0) > 5:
            flagged.append(col)
    if flagged:
        bullets.append(f"Columns with notable outliers (>5% of rows): {', '.join(flagged)}.")

    return bullets
