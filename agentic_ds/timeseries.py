"""Time series statistics: stationarity tests, decomposition, ACF/PACF,
residual whiteness, and a small-order ARIMA fit + forecast.

statsmodels only — already a dependency (Phase 5's GLM module). Column/
attribute names below (`result_object=True`, `.pvalue`, `.statistic`)
were verified against the actually-installed version: `adfuller`/`kpss`
are mid-deprecation on their old plain-tuple return, so this uses their
new attribute-based result objects instead of chasing tuple positions.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

ALPHA = 0.05
MAX_DIFFERENCING = 2


def synthetic_monthly_series(n: int = 144, seed: int = 42) -> pd.Series:
    """A clearly-labeled *synthetic* monthly series (trend + yearly
    seasonality + noise) for demoing this page — none of the app's other
    built-in datasets (iris/wine/breast_cancer/diabetes/titanic) have a
    time axis, and fabricating a "real" classic dataset (e.g. AirPassengers)
    from memory risks transcription errors in dozens of hardcoded numbers.
    This is honest about what it is instead.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 0.6 * t
    seasonal = 15 * np.sin(2 * np.pi * t / 12)
    noise = rng.normal(0, 4, n)
    values = 200 + trend + seasonal + noise
    index = pd.date_range("2013-01-01", periods=n, freq="MS")
    return pd.Series(values.round(2), index=index, name="value")


def check_stationarity(series: pd.Series) -> dict:
    """ADF (H0: unit root / non-stationary) and KPSS (H0: stationary) —
    run both, since neither alone is reliable and they check complementary
    null hypotheses. Verdict is "stationary"/"non-stationary" only when
    they agree; otherwise flagged inconclusive rather than guessing.
    """
    series = series.dropna()
    adf_result = adfuller(series, result_object=True)
    kpss_result = kpss(series, nlags="auto", result_object=True)

    adf_says_stationary = adf_result.pvalue < ALPHA
    kpss_says_stationary = kpss_result.pvalue > ALPHA

    if adf_says_stationary and kpss_says_stationary:
        verdict = "stationary"
    elif not adf_says_stationary and not kpss_says_stationary:
        verdict = "non-stationary"
    else:
        verdict = "inconclusive (ADF and KPSS disagree)"

    return {
        "verdict": verdict,
        "adf": {"statistic": round(float(adf_result.statistic), 4), "p_value": round(float(adf_result.pvalue), 4)},
        "kpss": {"statistic": round(float(kpss_result.statistic), 4), "p_value": round(float(kpss_result.pvalue), 4)},
    }


def suggest_differencing_order(series: pd.Series, max_d: int = MAX_DIFFERENCING) -> tuple[int, list[dict]]:
    """Differences the series repeatedly, checking ADF each time, until it
    looks stationary or `max_d` is hit. Returns (d, the ADF check at each step).
    """
    current = series.dropna()
    checks = []
    for d in range(max_d + 1):
        check = check_stationarity(current)
        checks.append({"d": d, **check})
        if check["verdict"] != "non-stationary":
            return d, checks
        current = current.diff().dropna()
    return max_d, checks


def decompose(series: pd.Series, period: int, model: str = "additive") -> pd.DataFrame:
    result = seasonal_decompose(series.dropna(), model=model, period=period)
    return pd.DataFrame({
        "observed": result.observed,
        "trend": result.trend,
        "seasonal": result.seasonal,
        "residual": result.resid,
    })


def autocorrelation(series: pd.Series, nlags: int = 24) -> pd.DataFrame:
    series = series.dropna()
    nlags = min(nlags, len(series) // 2 - 1)
    acf_values = acf(series, nlags=nlags)
    pacf_values = pacf(series, nlags=nlags)
    return pd.DataFrame({"lag": range(nlags + 1), "acf": acf_values, "pacf": pacf_values})


def whiteness_test(series: pd.Series, lags: int = 10) -> dict:
    """Ljung-Box: H0 is 'no autocorrelation' (the series/residuals are
    white noise). Used both on a raw series (is there structure to model
    at all?) and on ARIMA residuals (did the fit capture it all?).
    """
    series = series.dropna()
    lags = min(lags, len(series) // 2 - 1)
    result = acorr_ljungbox(series, lags=[lags], return_df=True)
    p_value = float(result["lb_pvalue"].iloc[0])
    return {
        "lags": lags,
        "statistic": round(float(result["lb_stat"].iloc[0]), 4),
        "p_value": round(p_value, 4),
        "white_noise": bool(p_value > ALPHA),
    }


def fit_arima(series: pd.Series, order: tuple[int, int, int], forecast_steps: int = 12) -> dict:
    series = series.dropna()
    fit_result = ARIMA(series, order=order).fit()

    coef_table = pd.DataFrame({
        "term": fit_result.params.index,
        "estimate": fit_result.params.values.round(4),
        "std_err": fit_result.bse.reindex(fit_result.params.index).values.round(4),
        "p_value": fit_result.pvalues.reindex(fit_result.params.index).values.round(6),
    })

    forecast = fit_result.get_forecast(steps=forecast_steps).summary_frame()
    residual_check = whiteness_test(fit_result.resid)

    return {
        "order": order,
        "n_obs": int(fit_result.nobs),
        "coefficients": coef_table.to_dict("records"),
        "aic": round(float(fit_result.aic), 2),
        "bic": round(float(fit_result.bic), 2),
        "forecast": forecast.reset_index().rename(columns={"index": "date"}).to_dict("records"),
        "residual_whiteness": residual_check,
    }


def interpret_stationarity(result: dict) -> str:
    sentence = f"ADF p = {result['adf']['p_value']}, KPSS p = {result['kpss']['p_value']} → {result['verdict']}."
    return sentence


def interpret_arima(summary: dict) -> str:
    order = summary["order"]
    sentence = f"ARIMA{order}, {summary['n_obs']} observations, AIC = {summary['aic']}."
    whiteness = summary["residual_whiteness"]
    if whiteness["white_noise"]:
        sentence += f" Residuals look like white noise (Ljung-Box p = {whiteness['p_value']}) — the model captured the autocorrelation structure."
    else:
        sentence += f" Residuals still show autocorrelation (Ljung-Box p = {whiteness['p_value']}) — consider a different order."
    return sentence
