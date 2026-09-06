"""Survival analysis: Kaplan-Meier curves, the log-rank test, and Cox
proportional hazards regression.

`lifelines` — verified installed and working (against its own bundled
`load_rossi()` dataset) before being committed to PLAN.md's Phase 8.
Column/attribute names below (`cph.summary`'s columns, `kmf.median_survival_time_`)
were confirmed against the actually-installed version.
"""

import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

ALPHA = 0.05


def kaplan_meier(durations, events, label: str = "all") -> dict:
    """Survival curve for one group. `events`: 1 = event observed
    (e.g. death/rearrest), 0 = censored (still alive/observed when data
    collection ended, or lost to follow-up)."""
    kmf = KaplanMeierFitter()
    kmf.fit(durations, events, label=label)

    curve = kmf.survival_function_.reset_index()
    curve.columns = ["time", "survival"]
    ci = kmf.confidence_interval_.reset_index()
    ci.columns = ["time", "ci_lower", "ci_upper"]
    curve = curve.merge(ci, on="time")

    median = kmf.median_survival_time_
    return {
        "label": label,
        "n": int(len(durations)),
        "n_events": int(sum(events)),
        "median_survival_time": None if median == float("inf") else round(float(median), 2),
        "curve": curve.to_dict("records"),
    }


def compare_groups(df: pd.DataFrame, duration_col: str, event_col: str, group_col: str) -> dict:
    """Kaplan-Meier curve per group + a log-rank test across exactly 2 groups."""
    data = df[[duration_col, event_col, group_col]].dropna()
    labels = sorted(data[group_col].unique())
    if len(labels) != 2:
        raise ValueError(f"compare_groups needs exactly 2 groups, got {len(labels)}: {labels}")

    curves = {}
    for label in labels:
        subset = data[data[group_col] == label]
        curves[str(label)] = kaplan_meier(subset[duration_col], subset[event_col], label=str(label))

    a = data[data[group_col] == labels[0]]
    b = data[data[group_col] == labels[1]]
    result = logrank_test(
        a[duration_col], b[duration_col], event_observed_A=a[event_col], event_observed_B=b[event_col]
    )

    return {
        "group_labels": [str(l) for l in labels],
        "curves": curves,
        "logrank": {
            "statistic": round(float(result.test_statistic), 4),
            "p_value": round(float(result.p_value), 6),
            "reject_null": bool(result.p_value < ALPHA),
            "group_labels": [str(l) for l in labels],
        },
    }


def fit_cox(df: pd.DataFrame, duration_col: str, event_col: str, covariates: list[str]) -> dict:
    """Cox proportional hazards: hazard ratios (exp(coef)) with real
    inference (p-values, CIs) — same "inference is the point" reasoning
    as Phase 5's GLM choice of statsmodels over sklearn's regressors.
    """
    data = df[[duration_col, event_col] + covariates].dropna()
    data = pd.get_dummies(data, columns=[c for c in covariates if not pd.api.types.is_numeric_dtype(data[c])], drop_first=True)

    cph = CoxPHFitter()
    cph.fit(data, duration_col=duration_col, event_col=event_col)

    summary = cph.summary.reset_index().rename(columns={"covariate": "term"})
    coef_table = summary[["term", "coef", "exp(coef)", "se(coef)", "p", "coef lower 95%", "coef upper 95%"]].rename(
        columns={"exp(coef)": "hazard_ratio", "se(coef)": "std_err", "p": "p_value"}
    )
    coef_table = coef_table.round(4)

    return {
        "n_obs": int(cph.event_observed.shape[0]),
        "n_events": int(cph.event_observed.sum()),
        "coefficients": coef_table.to_dict("records"),
        "concordance_index": round(float(cph.concordance_index_), 4),
        "aic": round(float(cph.AIC_partial_), 2),
        "log_likelihood": round(float(cph.log_likelihood_), 2),
    }


def interpret_km(result: dict) -> str:
    median = result["median_survival_time"]
    median_text = f"median survival time {median}" if median is not None else "median survival time not reached (more than half never had the event)"
    return f"{result['label']}: {result['n']} subjects, {result['n_events']} events observed, {median_text}."


def interpret_logrank(result: dict) -> str:
    verdict = "a statistically significant difference" if result["reject_null"] else "no statistically significant difference"
    return f"Log-rank test: p = {result['p_value']:.4g} → {verdict} in survival between {result['group_labels'][0]} and {result['group_labels'][1]}."


def interpret_cox(summary: dict) -> str:
    sig_terms = [c["term"] for c in summary["coefficients"] if c["p_value"] < ALPHA]
    sentence = (
        f"Cox model, {summary['n_obs']} subjects ({summary['n_events']} events), "
        f"concordance index = {summary['concordance_index']} (0.5 = no better than chance, 1.0 = perfect)."
    )
    if sig_terms:
        directions = []
        for c in summary["coefficients"]:
            if c["term"] in sig_terms:
                direction = "higher" if c["hazard_ratio"] > 1 else "lower"
                directions.append(f"{c['term']} ({direction} hazard, HR={c['hazard_ratio']})")
        sentence += f" Statistically significant (p < {ALPHA}): {', '.join(directions)}."
    else:
        sentence += " No covariate reached statistical significance."
    return sentence
