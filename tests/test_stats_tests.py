"""Runnable self-check for the hypothesis-testing catalog: verify each
result against scipy.stats called directly, and that the auto-recommend
logic actually switches branches when assumptions fail.

    python -m tests.test_stats_tests
"""

import numpy as np
from scipy import stats as scipy_stats

from agentic_ds import stats_tests as st

RNG = np.random.default_rng(42)


def test_normality_check_matches_shapiro():
    sample = RNG.normal(0, 1, 100)
    result = st.check_normality(sample)
    expected_stat, expected_p = scipy_stats.shapiro(sample)
    assert result["test"] == "shapiro"
    assert abs(result["statistic"] - expected_stat) < 1e-4
    assert abs(result["p_value"] - expected_p) < 1e-4


def test_two_groups_normal_picks_ttest_and_matches_scipy():
    a = RNG.normal(10, 2, 60)
    b = RNG.normal(10.5, 2, 60)
    result = st.compare_two_groups(np.concatenate([a, b]), ["a"] * 60 + ["b"] * 60)
    assert result["test"] in ("student_t", "welch_t")
    equal_var = result["test"] == "student_t"
    expected_stat, expected_p = scipy_stats.ttest_ind(a, b, equal_var=equal_var)
    assert abs(result["statistic"] - expected_stat) < 1e-4
    assert abs(result["p_value"] - expected_p) < 1e-4
    assert result["effect_size_name"] == "cohens_d"


def test_two_groups_non_normal_falls_back_to_mann_whitney():
    a = RNG.exponential(2.0, 80)  # deliberately skewed, not normal
    b = RNG.exponential(2.0, 80)
    result = st.compare_two_groups(np.concatenate([a, b]), ["a"] * 80 + ["b"] * 80)
    assert result["test"] == "mann_whitney_u"
    expected_stat, expected_p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    assert abs(result["statistic"] - expected_stat) < 1e-4
    assert abs(result["p_value"] - expected_p) < 1e-4
    assert result["effect_size"] is None  # no Cohen's d for a non-parametric test


def test_paired_normal_matches_ttest_rel():
    before = RNG.normal(50, 5, 40)
    after = before + RNG.normal(2, 1, 40)  # small normal shift
    result = st.compare_paired(before, after)
    assert result["test"] == "paired_t"
    expected_stat, expected_p = scipy_stats.ttest_rel(before, after)
    assert abs(result["statistic"] - expected_stat) < 1e-4
    assert abs(result["p_value"] - expected_p) < 1e-4


def test_many_groups_normal_equal_variance_matches_anova():
    groups = [RNG.normal(m, 3, 50) for m in (10, 12, 14)]
    values = np.concatenate(groups)
    labels = sum([[f"g{i}"] * len(g) for i, g in enumerate(groups)], [])
    result = st.compare_many_groups(values, labels)
    assert result["test"] == "one_way_anova"
    expected_stat, expected_p = scipy_stats.f_oneway(*groups)
    assert abs(result["statistic"] - expected_stat) < 1e-4
    assert abs(result["p_value"] - expected_p) < 1e-4
    assert result["reject_null"] is True  # means are 10/12/14 with tight spread — should differ


def test_many_groups_non_normal_falls_back_to_kruskal():
    groups = [RNG.exponential(2.0, 50) for _ in range(3)]
    values = np.concatenate(groups)
    labels = sum([[f"g{i}"] * len(g) for i, g in enumerate(groups)], [])
    result = st.compare_many_groups(values, labels)
    assert result["test"] == "kruskal_wallis"
    expected_stat, expected_p = scipy_stats.kruskal(*groups)
    assert abs(result["statistic"] - expected_stat) < 1e-4
    assert abs(result["p_value"] - expected_p) < 1e-4


def test_correlation_normal_picks_pearson():
    x = RNG.normal(0, 1, 100)
    y = 2 * x + RNG.normal(0, 0.5, 100)
    result = st.test_correlation(x, y)
    assert result["test"] == "pearson"
    expected_stat, expected_p = scipy_stats.pearsonr(x, y)
    assert abs(result["correlation"] - expected_stat) < 1e-4
    assert abs(result["p_value"] - expected_p) < 1e-4


def test_categorical_association_matches_chi_square():
    # A large, clearly-associated table — no small expected counts, so
    # chi-square (not Fisher's exact) should be picked.
    col_a = (["yes"] * 80 + ["no"] * 20) + (["yes"] * 20 + ["no"] * 80)
    col_b = ["A"] * 100 + ["B"] * 100
    result = st.test_categorical_association(col_a, col_b)
    assert result["test"] == "chi_square_independence"
    import pandas as pd

    table = pd.crosstab(pd.Series(col_a), pd.Series(col_b))
    expected_chi2, expected_p, _, _ = scipy_stats.chi2_contingency(table)
    assert abs(result["statistic"] - expected_chi2) < 1e-4
    assert abs(result["p_value"] - expected_p) < 1e-4
    assert result["reject_null"] is True


def test_categorical_association_small_counts_uses_fisher():
    # Classic small 2x2 table — expected counts will be under 5.
    col_a = ["yes"] * 3 + ["no"] * 1 + ["yes"] * 1 + ["no"] * 3
    col_b = ["A"] * 4 + ["B"] * 4
    result = st.test_categorical_association(col_a, col_b)
    assert result["test"] == "fisher_exact"


def test_one_sample_proportion_ztest_zero_when_matches_hypothesis():
    result = st.one_sample_proportion_ztest(count=50, nobs=100, value=0.5)
    assert abs(result["statistic"]) < 1e-9
    assert result["p_value"] > 0.99


if __name__ == "__main__":
    test_normality_check_matches_shapiro()
    test_two_groups_normal_picks_ttest_and_matches_scipy()
    test_two_groups_non_normal_falls_back_to_mann_whitney()
    test_paired_normal_matches_ttest_rel()
    test_many_groups_normal_equal_variance_matches_anova()
    test_many_groups_non_normal_falls_back_to_kruskal()
    test_correlation_normal_picks_pearson()
    test_categorical_association_matches_chi_square()
    test_categorical_association_small_counts_uses_fisher()
    test_one_sample_proportion_ztest_zero_when_matches_hypothesis()
    print("OK — all hypothesis-testing self-checks passed.")
