"""Runnable self-check for the survival module, against both synthetic
data with known ground truth and lifelines' own real bundled dataset.

    python -m tests.test_survival
"""

import numpy as np
import pandas as pd
from lifelines.datasets import load_rossi

from agentic_ds import survival as sv

RNG = np.random.default_rng(11)


def _simulate_cox_data(n: int, true_beta: float, baseline_hazard: float = 0.1, censor_time: float = 20.0):
    """Exponential-hazard Cox simulation: event time ~ Exp(baseline_hazard * exp(beta*x)),
    right-censored at a fixed time (a subject who hasn't had the event by
    then is recorded as censored, not observed to fail).
    """
    x = RNG.normal(0, 1, n)
    hazard = baseline_hazard * np.exp(true_beta * x)
    event_time = RNG.exponential(1 / hazard)
    duration = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)
    return pd.DataFrame({"x": x, "duration": duration, "event": event})


def test_kaplan_meier_survival_is_monotonic_non_increasing():
    df = load_rossi()
    result = sv.kaplan_meier(df["week"], df["arrest"], label="all")
    survival = [row["survival"] for row in result["curve"]]
    assert all(survival[i] >= survival[i + 1] for i in range(len(survival) - 1))
    assert survival[0] == 1.0  # everyone alive at time 0
    assert result["n"] == 432 and result["n_events"] == 114  # known shape of the real dataset


def test_logrank_distinguishes_genuinely_different_groups():
    # Group B has 3x the hazard of group A -> genuinely different survival.
    n = 300
    duration_a = RNG.exponential(20, n)
    duration_b = RNG.exponential(20 / 3, n)
    df = pd.DataFrame({
        "duration": np.concatenate([duration_a, duration_b]),
        "event": np.ones(2 * n, dtype=int),
        "group": ["A"] * n + ["B"] * n,
    })
    result = sv.compare_groups(df, "duration", "event", "group")
    assert result["logrank"]["reject_null"] is True
    assert result["logrank"]["p_value"] < 0.001  # 3x hazard difference should be obvious at n=300/group


def test_logrank_no_difference_when_groups_are_identical():
    n = 200
    duration = RNG.exponential(20, 2 * n)
    df = pd.DataFrame({"duration": duration, "event": np.ones(2 * n, dtype=int), "group": ["A"] * n + ["B"] * n})
    result = sv.compare_groups(df, "duration", "event", "group")
    assert result["logrank"]["reject_null"] is False


def test_cox_recovers_known_coefficient():
    df = _simulate_cox_data(n=800, true_beta=0.8)
    summary = sv.fit_cox(df, "duration", "event", ["x"])
    coef = next(c for c in summary["coefficients"] if c["term"] == "x")
    assert abs(coef["coef"] - 0.8) < 0.15
    assert coef["hazard_ratio"] > 1  # positive beta -> higher hazard -> HR > 1
    assert coef["p_value"] < 0.05  # should be a clearly detectable effect at n=800
    assert 0.5 <= summary["concordance_index"] <= 1.0


def test_cox_on_real_rossi_data_matches_known_direction():
    # Sanity check against the well-known result on this famous dataset:
    # financial aid after release ('fin') reduces the hazard of rearrest.
    df = load_rossi()
    summary = sv.fit_cox(df, "week", "arrest", ["fin", "age", "race", "wexp", "mar", "paro", "prio"])
    fin_coef = next(c for c in summary["coefficients"] if c["term"] == "fin")
    assert fin_coef["hazard_ratio"] < 1  # aid lowers hazard
    assert fin_coef["p_value"] < 0.05


if __name__ == "__main__":
    test_kaplan_meier_survival_is_monotonic_non_increasing()
    test_logrank_distinguishes_genuinely_different_groups()
    test_logrank_no_difference_when_groups_are_identical()
    test_cox_recovers_known_coefficient()
    test_cox_on_real_rossi_data_matches_known_direction()
    print("OK — all survival-analysis self-checks passed.")
