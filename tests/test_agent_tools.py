"""Runnable self-check for Phase 9a's Tool layer: every tool has a valid
schema, and a representative tool from each category is actually callable
against a real small DataFrame (including one call with bad input, to
confirm errors come back as a JSON `{"error": ...}` string instead of
raising and killing a tool-calling loop).

    python -m tests.test_agent_tools
"""

import json

import numpy as np
import pandas as pd

from agentic_ds.agent_tools import build_tools

RNG = np.random.default_rng(0)
N = 200


def _sample_df() -> pd.DataFrame:
    x1 = RNG.normal(0, 1, N)
    x2 = RNG.normal(5, 2, N)
    group2 = np.where(RNG.random(N) < 0.5, "A", "B")
    group3 = RNG.choice(["low", "mid", "high"], N)
    duration = RNG.exponential(20, N)
    event = RNG.integers(0, 2, N)
    count_target = RNG.poisson(5, N)
    binary_target = RNG.integers(0, 2, N)
    return pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "group2": group2,
        "group3": group3,
        "duration": duration,
        "event": event,
        "count_target": count_target,
        "binary_target": binary_target,
    })


def test_tool_count_without_model():
    tools = build_tools(_sample_df())
    names = {t.name for t in tools}
    assert "smart_sql_query" not in names  # no model passed -> SQL tool omitted
    assert len(tools) == 26


def test_tool_count_with_model():
    tools = build_tools(_sample_df(), model=object())  # any truthy stand-in — never actually called here
    names = {t.name for t in tools}
    assert "smart_sql_query" in names
    assert len(tools) == 27


def test_every_tool_has_a_valid_schema():
    tools = build_tools(_sample_df())
    for t in tools:
        assert t.name, "tool with no name"
        assert t.description and len(t.description) > 10, f"{t.name} has no real description"
        assert isinstance(t.args, dict)


def _invoke(tools_by_name: dict, name: str, args: dict) -> dict:
    return json.loads(tools_by_name[name].invoke(args))


def test_eda_tools_callable():
    tools = {t.name: t for t in build_tools(_sample_df())}
    schema = _invoke(tools, "get_schema_overview", {})
    assert isinstance(schema, list) and len(schema) == 8  # 8 columns in the fixture

    outliers = _invoke(tools, "detect_column_outliers", {"column": "x1"})
    assert "outlier_count" in outliers


def test_hypothesis_test_tool_callable_and_matches_direct_call():
    df = _sample_df()
    tools = {t.name: t for t in build_tools(df)}
    from agentic_ds import stats_tests as sx

    via_tool = _invoke(tools, "compare_two_groups", {"numeric_column": "x1", "group_column": "group2"})
    direct = sx.compare_two_groups(df["x1"], df["group2"])
    assert via_tool["test"] == direct["test"]
    assert via_tool["p_value"] == direct["p_value"]


def test_glm_tool_callable():
    tools = {t.name: t for t in build_tools(_sample_df())}
    suggestion = _invoke(tools, "suggest_glm_family", {"target_column": "binary_target"})
    assert suggestion["family"] == "binomial"

    fit = _invoke(tools, "fit_glm_model", {"target_column": "binary_target", "predictor_columns": ["x1", "x2"], "family": "binomial"})
    assert "coefficients" in fit and "aic" in fit


def test_timeseries_tool_callable():
    tools = {t.name: t for t in build_tools(_sample_df())}
    result = _invoke(tools, "check_series_stationarity", {"column": "x1"})
    assert result["verdict"] in ("stationary", "non-stationary", "inconclusive (ADF and KPSS disagree)")


def test_survival_tool_callable():
    tools = {t.name: t for t in build_tools(_sample_df())}
    result = _invoke(tools, "compute_kaplan_meier", {"duration_column": "duration", "event_column": "event"})
    assert result["n"] == N


def test_bayesian_tool_callable():
    # Slower (MCMC) — one call is enough to confirm the wrapping works;
    # agentic_ds/bayesian.py's own tests already cover correctness.
    tools = {t.name: t for t in build_tools(_sample_df())}
    result = _invoke(tools, "fit_bayesian_model", {"target_column": "binary_target", "predictor_columns": ["x1"], "family": "logistic"})
    assert "coefficients" in result and "max_rhat" in result


def test_bad_column_name_returns_error_json_instead_of_raising():
    tools = {t.name: t for t in build_tools(_sample_df())}
    result = _invoke(tools, "compare_two_groups", {"numeric_column": "does_not_exist", "group_column": "group2"})
    assert "error" in result


def test_wrong_group_count_returns_error_json_instead_of_raising():
    tools = {t.name: t for t in build_tools(_sample_df())}
    # group3 has 3 levels; compare_two_groups needs exactly 2 — should error, not raise.
    result = _invoke(tools, "compare_two_groups", {"numeric_column": "x1", "group_column": "group3"})
    assert "error" in result


if __name__ == "__main__":
    test_tool_count_without_model()
    test_tool_count_with_model()
    test_every_tool_has_a_valid_schema()
    test_eda_tools_callable()
    test_hypothesis_test_tool_callable_and_matches_direct_call()
    test_glm_tool_callable()
    test_timeseries_tool_callable()
    test_survival_tool_callable()
    test_bayesian_tool_callable()
    test_bad_column_name_returns_error_json_instead_of_raising()
    test_wrong_group_count_returns_error_json_instead_of_raising()
    print("OK — all agent-tools self-checks passed.")
