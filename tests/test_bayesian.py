"""Runnable self-check for the Bayesian module: posterior means recover
known synthetic coefficients within a generous tolerance (MCMC is
stochastic — fixed seed, wide tolerance, not exact-match), and
convergence diagnostics are actually reported.

    python -m tests.test_bayesian
"""

import numpy as np
import pandas as pd

from agentic_ds import bayesian as bys

RNG = np.random.default_rng(7)
DRAWS, TUNE, CHAINS = 400, 400, 2  # low but not the UI default — keeps the test quick


def test_suggest_family_maps_glm_shapes():
    assert bys.suggest_family(pd.Series(RNG.normal(0, 1, 100)))[0] == "linear"
    assert bys.suggest_family(pd.Series(RNG.integers(0, 2, 100)))[0] == "logistic"
    assert bys.suggest_family(pd.Series(RNG.poisson(5, 100)))[0] == "poisson"


def test_linear_recovers_known_coefficients():
    n = 300
    x1 = RNG.normal(0, 1, n)
    true_intercept, true_beta = 2.0, 1.5
    y = true_intercept + true_beta * x1 + RNG.normal(0, 1, n)
    df = pd.DataFrame({"x1": x1, "y": y})

    idata = bys.fit_bayesian(df, "y", ["x1"], "linear", draws=DRAWS, tune=TUNE, chains=CHAINS, seed=7)
    summary = bys.summarize(idata)
    coefs = {c["term"]: c["mean"] for c in summary["coefficients"]}

    assert abs(coefs["intercept"] - true_intercept) < 0.3
    assert abs(coefs["beta[x1]"] - true_beta) < 0.3
    assert "max_rhat" in summary and summary["max_rhat"] > 0  # actually reported, not dropped
    assert summary["converged"] is True


def test_logistic_recovers_known_coefficients():
    n = 400
    x1 = RNG.normal(0, 1, n)
    true_intercept, true_beta = 0.5, 1.2
    p = 1 / (1 + np.exp(-(true_intercept + true_beta * x1)))
    y = RNG.binomial(1, p)
    df = pd.DataFrame({"x1": x1, "y": y})

    idata = bys.fit_bayesian(df, "y", ["x1"], "logistic", draws=DRAWS, tune=TUNE, chains=CHAINS, seed=7)
    summary = bys.summarize(idata)
    coefs = {c["term"]: c["mean"] for c in summary["coefficients"]}

    assert abs(coefs["intercept"] - true_intercept) < 0.5
    assert abs(coefs["beta[x1]"] - true_beta) < 0.5
    assert "sigma" not in coefs  # no dispersion parameter for a Bernoulli likelihood


def test_poisson_recovers_known_coefficients():
    n = 300
    x1 = RNG.normal(0, 1, n)
    true_intercept, true_beta = 0.7, 0.5
    mu = np.exp(true_intercept + true_beta * x1)
    y = RNG.poisson(mu)
    df = pd.DataFrame({"x1": x1, "y": y})

    idata = bys.fit_bayesian(df, "y", ["x1"], "poisson", draws=DRAWS, tune=TUNE, chains=CHAINS, seed=7)
    summary = bys.summarize(idata)
    coefs = {c["term"]: c["mean"] for c in summary["coefficients"]}

    assert abs(coefs["intercept"] - true_intercept) < 0.2
    assert abs(coefs["beta[x1]"] - true_beta) < 0.2


def test_posterior_predictive_samples_are_plausible():
    n = 200
    x1 = RNG.normal(0, 1, n)
    y = 3 + 2 * x1 + RNG.normal(0, 1, n)
    df = pd.DataFrame({"x1": x1, "y": y})
    idata = bys.fit_bayesian(df, "y", ["x1"], "linear", draws=DRAWS, tune=TUNE, chains=CHAINS, seed=7)
    samples = bys.posterior_predictive_samples(idata, n=200, seed=7)
    assert len(samples) == 200
    assert abs(samples.mean() - y.mean()) < 1.5  # simulated outcomes in the right ballpark


def test_interpret_flags_non_convergence():
    fake_summary = {
        "coefficients": [{"term": "intercept", "hdi_lower": -0.1, "hdi_upper": 0.1}],
        "hdi_prob": 0.94,
        "max_rhat": 1.05,
        "converged": False,
        "min_ess_bulk": 50.0,
    }
    sentence = bys.interpret(fake_summary)
    assert "WARNING" in sentence
    assert "converged" in sentence.lower()


if __name__ == "__main__":
    test_suggest_family_maps_glm_shapes()
    test_linear_recovers_known_coefficients()
    test_logistic_recovers_known_coefficients()
    test_poisson_recovers_known_coefficients()
    test_posterior_predictive_samples_are_plausible()
    test_interpret_flags_non_convergence()
    print("OK — all Bayesian self-checks passed.")
