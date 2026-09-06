"""Runnable self-check for the time-series module.

    python -m tests.test_timeseries
"""

import numpy as np
import pandas as pd

from agentic_ds import timeseries as ts

RNG = np.random.default_rng(3)


def test_synthetic_series_shape():
    series = ts.synthetic_monthly_series(n=144, seed=1)
    assert len(series) == 144
    assert isinstance(series.index, pd.DatetimeIndex)


def test_trending_series_is_non_stationary():
    n = 200
    trend = np.arange(n) * 0.5
    series = pd.Series(trend + RNG.normal(0, 1, n))
    result = ts.check_stationarity(series)
    assert result["verdict"] == "non-stationary"


def test_white_noise_is_stationary():
    series = pd.Series(RNG.normal(0, 1, 300))
    result = ts.check_stationarity(series)
    assert result["verdict"] == "stationary"


def test_differencing_a_trend_reaches_stationarity():
    n = 200
    trend = np.arange(n) * 0.5
    series = pd.Series(trend + RNG.normal(0, 1, n))
    d, checks = ts.suggest_differencing_order(series, max_d=2)
    assert d >= 1  # needed at least one difference
    assert checks[0]["verdict"] == "non-stationary"  # d=0 was indeed non-stationary
    assert checks[-1]["verdict"] != "non-stationary"  # final check improved


def test_decompose_recovers_trend_direction():
    series = ts.synthetic_monthly_series(n=144, seed=2)
    components = ts.decompose(series, period=12)
    trend = components["trend"].dropna()
    assert trend.iloc[-1] > trend.iloc[0]  # synthetic series has a positive trend by construction


def test_whiteness_test_distinguishes_white_noise_from_autocorrelated():
    white = pd.Series(RNG.normal(0, 1, 300))
    assert ts.whiteness_test(white)["white_noise"] is True

    n = 300
    autocorrelated = np.zeros(n)
    for i in range(1, n):
        autocorrelated[i] = 0.9 * autocorrelated[i - 1] + RNG.normal(0, 1)
    assert ts.whiteness_test(pd.Series(autocorrelated))["white_noise"] is False


def test_arima_fit_and_forecast_shape():
    series = ts.synthetic_monthly_series(n=144, seed=4)
    summary = ts.fit_arima(series, order=(1, 1, 1), forecast_steps=6)
    assert len(summary["forecast"]) == 6
    assert "aic" in summary and summary["aic"] > 0
    assert summary["residual_whiteness"]["p_value"] >= 0


if __name__ == "__main__":
    test_synthetic_series_shape()
    test_trending_series_is_non_stationary()
    test_white_noise_is_stationary()
    test_differencing_a_trend_reaches_stationarity()
    test_decompose_recovers_trend_direction()
    test_whiteness_test_distinguishes_white_noise_from_autocorrelated()
    test_arima_fit_and_forecast_shape()
    print("OK — all time-series self-checks passed.")
