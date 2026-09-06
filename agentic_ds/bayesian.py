"""Bayesian counterparts to the three most common GLM families — linear,
logistic, Poisson — via PyMC, with weakly-informative default priors
(Normal(0, 10) on coefficients, HalfNormal(5) on scale terms) so getting a
first result never requires specifying priors.

pymc + arviz — see PLAN.md "Modules and libraries for Phases 4-6" for the
install/timing numbers verified before this was committed to the plan.
Column names below (`hdi94_lb` etc.) come from arviz 1.3's `az.summary`;
confirmed against the actual installed version, not assumed, since arviz
renamed this API (`hdi_prob` -> `ci_prob`/`ci_kind`) after most online
examples were written.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

from agentic_ds.glm import detect_family

FAMILIES = ["linear", "logistic", "poisson"]

FAMILY_DESCRIPTIONS = {
    "linear": "Linear (Normal likelihood) — continuous, roughly symmetric target",
    "logistic": "Logistic (Bernoulli likelihood) — binary (0/1) target",
    "poisson": "Poisson (log link) — non-negative integer count target",
}

RHAT_THRESHOLD = 1.01
HDI_PROB = 0.94


def suggest_family(y: pd.Series) -> tuple[str, str]:
    """Reuses the GLM page's shape detection, then maps onto the fixed
    3-family Bayesian set. Gamma/Tweedie/etc have no counterpart here —
    falls back to linear with a note, rather than silently picking one.
    """
    glm_family, reason = detect_family(y)
    mapping = {"gaussian": "linear", "binomial": "logistic", "poisson": "poisson"}
    if glm_family in mapping:
        return mapping[glm_family], reason
    return "linear", f"{reason} (no Bayesian family here for '{glm_family}' — defaulting to linear)"


def _design_matrix(df: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    return pd.get_dummies(df[predictors], drop_first=True).astype(float)


def fit_bayesian(
    df: pd.DataFrame,
    target: str,
    predictors: list[str],
    family: str,
    draws: int = 500,
    tune: int = 500,
    chains: int = 2,
    seed: int = 42,
):
    """Returns an arviz InferenceData with both the posterior and posterior
    predictive samples attached.
    """
    data = df[[target] + predictors].dropna()
    X = _design_matrix(data, predictors)
    y = data[target].to_numpy()
    feature_names = X.columns.tolist()
    X_values = X.to_numpy()

    with pm.Model(coords={"feature": feature_names}):
        intercept = pm.Normal("intercept", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10, dims="feature")
        eta = intercept + pm.math.dot(X_values, beta)

        if family == "linear":
            sigma = pm.HalfNormal("sigma", sigma=5)
            pm.Normal("obs", mu=eta, sigma=sigma, observed=y)
        elif family == "logistic":
            pm.Bernoulli("obs", p=pm.math.sigmoid(eta), observed=y)
        elif family == "poisson":
            pm.Poisson("obs", mu=pm.math.exp(eta), observed=y)
        else:
            raise ValueError(f"Unknown family '{family}'. Choose one of: {FAMILIES}")

        idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed, progressbar=False)
        pm.sample_posterior_predictive(idata, random_seed=seed, progressbar=False, extend_inferencedata=True)

    return idata


def summarize(idata) -> dict:
    """Coefficient table (mean, sd, 94% HDI) + convergence diagnostics, in
    a plain-dict shape the UI and narrative layer can both consume.
    """
    raw = az.summary(idata, var_names=["intercept", "beta", "sigma"], ci_prob=HDI_PROB, ci_kind="hdi", filter_vars="like")
    raw = raw.rename(columns={f"hdi{int(HDI_PROB * 100)}_lb": "hdi_lower", f"hdi{int(HDI_PROB * 100)}_ub": "hdi_upper"})
    raw = raw.reset_index().rename(columns={"index": "term"})

    coefficients = raw[["term", "mean", "sd", "hdi_lower", "hdi_upper", "r_hat", "ess_bulk"]].to_dict("records")
    max_rhat = float(raw["r_hat"].max())

    return {
        "coefficients": coefficients,
        "hdi_prob": HDI_PROB,
        "max_rhat": round(max_rhat, 4),
        "converged": bool(max_rhat < RHAT_THRESHOLD),
        "min_ess_bulk": round(float(raw["ess_bulk"].min()), 1),
    }


def posterior_predictive_samples(idata, n: int = 500, seed: int = 42) -> np.ndarray:
    """A flat sample of simulated outcomes for a posterior-predictive-check
    plot (observed vs. simulated distribution)."""
    rng = np.random.default_rng(seed)
    values = idata.posterior_predictive["obs"].to_numpy().reshape(-1)
    if len(values) > n:
        values = rng.choice(values, size=n, replace=False)
    return values


def interpret(summary: dict) -> str:
    """Rule-based, always-available floor — no LLM required."""
    sentence = f"{summary['max_rhat']} max r-hat, {summary['min_ess_bulk']} min effective sample size."
    if not summary["converged"]:
        sentence += f" WARNING: r-hat > {RHAT_THRESHOLD} — sampling may not have converged, treat results cautiously."
    non_predictor_terms = {"sigma"}
    credible_terms = [
        c["term"]
        for c in summary["coefficients"]
        if c["term"] not in non_predictor_terms and not (c["hdi_lower"] < 0 < c["hdi_upper"])
    ]
    if credible_terms:
        sentence += f" {int(summary['hdi_prob'] * 100)}% HDI excludes zero for: {', '.join(credible_terms)}."
    else:
        sentence += f" No term's {int(summary['hdi_prob'] * 100)}% HDI excludes zero."
    return sentence
