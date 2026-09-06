"""Statistical hypothesis testing: assumption checks, a fixed test catalog,
and an auto-recommendation layer that picks the right test the way a
statistician would — check assumptions first, fall back to the
non-parametric equivalent when they fail.

scipy.stats only (see PLAN.md "Modules and libraries for Phases 4-6" for
why) — every function returns a plain dict so the UI and the narrative
layer can both consume it without re-parsing anything.
"""

import math

import pandas as pd
from scipy import stats

ALPHA = 0.05


def _clean(sample) -> pd.Series:
    return pd.Series(sample).dropna()


# ─────────────────────────── assumption checks ───────────────────────────


def check_normality(sample) -> dict:
    """Shapiro-Wilk for n < 5000 (the usual recommendation — it loses power
    and can flag trivial deviations as significant on very large samples),
    D'Agostino-Pearson otherwise.
    """
    sample = _clean(sample)
    if len(sample) < 3:
        return {"test": "normality", "n": len(sample), "normal": None, "note": "need at least 3 values"}
    if len(sample) < 5000:
        stat, p = stats.shapiro(sample)
        test_name = "shapiro"
    else:
        stat, p = stats.normaltest(sample)
        test_name = "dagostino_pearson"
    return {
        "test": test_name,
        "n": int(len(sample)),
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "normal": bool(p > ALPHA),
    }


def check_variance_homogeneity(*samples) -> dict:
    """Levene's test — robust to non-normality, the standard default over Bartlett."""
    samples = [_clean(s) for s in samples]
    stat, p = stats.levene(*samples)
    return {
        "test": "levene",
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "equal_variance": bool(p > ALPHA),
    }


# ─────────────────────────── effect sizes ───────────────────────────


def cohens_d(a, b) -> float:
    a, b = _clean(a), _clean(b)
    pooled_std = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return round(float((a.mean() - b.mean()) / pooled_std), 4) if pooled_std > 0 else 0.0


def eta_squared(*groups) -> float:
    groups = [_clean(g) for g in groups]
    all_values = pd.concat(groups)
    grand_mean = all_values.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((all_values - grand_mean) ** 2).sum()
    return round(float(ss_between / ss_total), 4) if ss_total > 0 else 0.0


def cramers_v(contingency: pd.DataFrame) -> float:
    chi2, _, _, _ = stats.chi2_contingency(contingency)
    n = contingency.to_numpy().sum()
    r, c = contingency.shape
    return round(float(math.sqrt((chi2 / n) / (min(r, c) - 1))), 4) if min(r, c) > 1 else 0.0


# ─────────────────────────── two-group comparison ───────────────────────────


def compare_two_groups(values, groups) -> dict:
    """Auto-recommends Welch's t-test (assumptions hold) or Mann-Whitney U
    (they don't). `groups` must have exactly 2 distinct labels.
    """
    df = pd.DataFrame({"value": values, "group": groups}).dropna()
    labels = sorted(df["group"].unique())
    if len(labels) != 2:
        raise ValueError(f"compare_two_groups needs exactly 2 groups, got {len(labels)}: {labels}")
    a = df.loc[df["group"] == labels[0], "value"]
    b = df.loc[df["group"] == labels[1], "value"]

    normality = {str(labels[0]): check_normality(a), str(labels[1]): check_normality(b)}
    both_normal = all(r["normal"] for r in normality.values() if r["normal"] is not None)
    variance = check_variance_homogeneity(a, b)

    if both_normal:
        stat, p = stats.ttest_ind(a, b, equal_var=variance["equal_variance"])
        test_name = "student_t" if variance["equal_variance"] else "welch_t"
        effect = cohens_d(a, b)
        effect_name = "cohens_d"
    else:
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        test_name = "mann_whitney_u"
        effect = None
        effect_name = None

    return {
        "test": test_name,
        "group_labels": [str(l) for l in labels],
        "group_sizes": [int(len(a)), int(len(b))],
        "group_means": [round(float(a.mean()), 4), round(float(b.mean()), 4)],
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "reject_null": bool(p < ALPHA),
        "effect_size": effect,
        "effect_size_name": effect_name,
        "assumptions": {"normality": normality, "variance_homogeneity": variance},
    }


def compare_paired(a, b) -> dict:
    """Auto-recommends paired t-test (differences roughly normal) or
    Wilcoxon signed-rank otherwise. `a` and `b` must be the same length
    (matched pairs, e.g. before/after on the same subjects).
    """
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    diffs = df["a"] - df["b"]
    normality = check_normality(diffs)

    if normality["normal"]:
        stat, p = stats.ttest_rel(df["a"], df["b"])
        test_name = "paired_t"
        effect = round(float(diffs.mean() / diffs.std(ddof=1)), 4) if diffs.std(ddof=1) > 0 else 0.0
    else:
        stat, p = stats.wilcoxon(df["a"], df["b"])
        test_name = "wilcoxon_signed_rank"
        effect = None

    return {
        "test": test_name,
        "n_pairs": int(len(df)),
        "mean_difference": round(float(diffs.mean()), 4),
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "reject_null": bool(p < ALPHA),
        "effect_size": effect,
        "effect_size_name": "cohens_d_paired" if effect is not None else None,
        "assumptions": {"normality_of_differences": normality},
    }


# ─────────────────────────── many-group comparison ───────────────────────────


def compare_many_groups(values, groups) -> dict:
    """Auto-recommends one-way ANOVA (assumptions hold) or Kruskal-Wallis
    otherwise. `groups` must have 3+ distinct labels (use compare_two_groups
    for exactly 2 — ANOVA/Kruskal-Wallis only tell you *some* group differs,
    not which).
    """
    df = pd.DataFrame({"value": values, "group": groups}).dropna()
    labels = sorted(df["group"].unique())
    if len(labels) < 3:
        raise ValueError(f"compare_many_groups needs 3+ groups, got {len(labels)}: {labels}")
    samples = [df.loc[df["group"] == label, "value"] for label in labels]

    normality = {str(label): check_normality(s) for label, s in zip(labels, samples)}
    both_normal = all(r["normal"] for r in normality.values() if r["normal"] is not None)
    variance = check_variance_homogeneity(*samples)

    if both_normal and variance["equal_variance"]:
        stat, p = stats.f_oneway(*samples)
        test_name = "one_way_anova"
        effect = eta_squared(*samples)
        effect_name = "eta_squared"
    else:
        stat, p = stats.kruskal(*samples)
        test_name = "kruskal_wallis"
        effect = None
        effect_name = None

    return {
        "test": test_name,
        "group_labels": [str(l) for l in labels],
        "group_sizes": [int(len(s)) for s in samples],
        "group_means": [round(float(s.mean()), 4) for s in samples],
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "reject_null": bool(p < ALPHA),
        "effect_size": effect,
        "effect_size_name": effect_name,
        "assumptions": {"normality": normality, "variance_homogeneity": variance},
    }


# ─────────────────────────── correlation ───────────────────────────


def test_correlation(x, y) -> dict:
    """Auto-recommends Pearson (both variables roughly normal) or Spearman
    (rank-based, no normality assumption) otherwise.
    """
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    normality = {"x": check_normality(df["x"]), "y": check_normality(df["y"])}
    both_normal = all(r["normal"] for r in normality.values() if r["normal"] is not None)

    if both_normal:
        stat, p = stats.pearsonr(df["x"], df["y"])
        test_name = "pearson"
    else:
        stat, p = stats.spearmanr(df["x"], df["y"])
        test_name = "spearman"

    return {
        "test": test_name,
        "n": int(len(df)),
        "correlation": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "reject_null": bool(p < ALPHA),
        "assumptions": {"normality": normality},
    }


# ─────────────────────────── categorical association ───────────────────────────


def test_categorical_association(col_a, col_b) -> dict:
    """Chi-square test of independence, or Fisher's exact for a 2x2 table
    with any expected cell count under 5 (chi-square's approximation
    breaks down there — Fisher's is exact regardless of sample size).
    """
    df = pd.DataFrame({"a": col_a, "b": col_b}).dropna()
    table = pd.crosstab(df["a"], df["b"])

    chi2, p, dof, expected = stats.chi2_contingency(table)
    if table.shape == (2, 2) and (expected < 5).any():
        _, p_exact = stats.fisher_exact(table)
        return {
            "test": "fisher_exact",
            "contingency_table": table.to_dict(),
            "p_value": round(float(p_exact), 6),
            "reject_null": bool(p_exact < ALPHA),
            "note": "chi-square's approximation is unreliable here (an expected cell count < 5), used Fisher's exact instead",
        }

    return {
        "test": "chi_square_independence",
        "contingency_table": table.to_dict(),
        "statistic": round(float(chi2), 4),
        "degrees_of_freedom": int(dof),
        "p_value": round(float(p), 6),
        "reject_null": bool(p < ALPHA),
        "effect_size": cramers_v(table),
        "effect_size_name": "cramers_v",
    }


# ─────────────────────────── one-sample tests ───────────────────────────


def one_sample_test(sample, popmean: float) -> dict:
    """Auto-recommends a one-sample t-test (roughly normal) or Wilcoxon
    signed-rank against `popmean` otherwise.
    """
    sample = _clean(sample)
    normality = check_normality(sample)

    if normality["normal"]:
        stat, p = stats.ttest_1samp(sample, popmean)
        test_name = "one_sample_t"
    else:
        stat, p = stats.wilcoxon(sample - popmean)
        test_name = "wilcoxon_one_sample"

    return {
        "test": test_name,
        "n": int(len(sample)),
        "sample_mean": round(float(sample.mean()), 4),
        "hypothesized_value": popmean,
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "reject_null": bool(p < ALPHA),
        "assumptions": {"normality": normality},
    }


def one_sample_proportion_ztest(count: int, nobs: int, value: float) -> dict:
    phat = count / nobs
    se = math.sqrt(value * (1 - value) / nobs)
    z = (phat - value) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {
        "test": "one_sample_proportion_z",
        "sample_proportion": round(phat, 4),
        "hypothesized_value": value,
        "n": nobs,
        "statistic": round(float(z), 4),
        "p_value": round(float(p), 6),
        "reject_null": bool(p < ALPHA),
    }


def interpret(result: dict) -> str:
    """Rule-based, always-available plain-English verdict — no LLM required.
    `llm.generate_narrative` can produce a richer version on top of this
    when an API key is set; this is the floor, not the ceiling."""
    test_label = result["test"].replace("_", " ")
    p = result["p_value"]
    verdict = "statistically significant" if result.get("reject_null") else "not statistically significant"
    sentence = f"{test_label}: p = {p:.4g} → {verdict} at α = {ALPHA}."
    effect = result.get("effect_size")
    if effect is not None:
        sentence += f" Effect size ({result['effect_size_name']}) = {effect}."
    if result.get("note"):
        sentence += f" ({result['note']})"
    return sentence


def two_sample_proportion_ztest(count1: int, nobs1: int, count2: int, nobs2: int) -> dict:
    p1, p2 = count1 / nobs1, count2 / nobs2
    p_pool = (count1 + count2) / (nobs1 + nobs2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / nobs1 + 1 / nobs2))
    z = (p1 - p2) / se if se > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {
        "test": "two_sample_proportion_z",
        "proportions": [round(p1, 4), round(p2, 4)],
        "ns": [nobs1, nobs2],
        "statistic": round(float(z), 4),
        "p_value": round(float(p), 6),
        "reject_null": bool(p < ALPHA),
    }
