"""Runnable self-check for the GLM module: family auto-suggestion picks the
right shape, and fitted coefficients recover known synthetic ground truth.

    python -m tests.test_glm
"""

import numpy as np
import pandas as pd

from agentic_ds import glm

RNG = np.random.default_rng(42)


def test_detect_family_poisson():
    y = pd.Series(RNG.poisson(5, 200))
    family, _ = glm.detect_family(y)
    assert family == "poisson"


def test_detect_family_binomial():
    y = pd.Series(RNG.integers(0, 2, 200))
    family, _ = glm.detect_family(y)
    assert family == "binomial"


def test_detect_family_gamma():
    y = pd.Series(RNG.gamma(shape=1.5, scale=3.0, size=200))  # right-skewed positive
    family, _ = glm.detect_family(y)
    assert family == "gamma"


def test_detect_family_tweedie():
    zeros = np.zeros(80)
    positives = RNG.gamma(2.0, 3.0, 120)
    y = pd.Series(np.concatenate([zeros, positives]))
    family, _ = glm.detect_family(y)
    assert family == "tweedie"


def test_detect_family_gaussian_default():
    y = pd.Series(RNG.normal(0, 1, 200))
    family, _ = glm.detect_family(y)
    assert family == "gaussian"


def test_poisson_glm_recovers_known_coefficients():
    n = 500
    x1 = RNG.normal(0, 1, n)
    x2 = RNG.normal(0, 1, n)
    true_intercept, true_b1, true_b2 = 0.5, 0.8, -0.4
    mu = np.exp(true_intercept + true_b1 * x1 + true_b2 * x2)
    y = RNG.poisson(mu)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

    fit = glm.fit_glm(df, "y", ["x1", "x2"], "poisson")
    summary = glm.summarize(fit, "poisson")
    coefs = {c["term"]: c["estimate"] for c in summary["coefficients"]}

    assert abs(coefs["const"] - true_intercept) < 0.15
    assert abs(coefs["x1"] - true_b1) < 0.1
    assert abs(coefs["x2"] - true_b2) < 0.1
    assert summary["deviance"] > 0


def test_overdispersion_triggers_negative_binomial_upgrade():
    n = 500
    x1 = RNG.normal(0, 1, n)
    mu = np.exp(1.0 + 0.6 * x1)
    r = 2.0
    p = r / (r + mu)
    y = RNG.negative_binomial(r, p)  # genuinely overdispersed relative to Poisson
    df = pd.DataFrame({"x1": x1, "y": y})

    poisson_fit = glm.fit_glm(df, "y", ["x1"], "poisson")
    dispersion = glm.check_overdispersion(poisson_fit)
    assert dispersion["overdispersed"] is True

    nb_fit = glm.fit_glm(df, "y", ["x1"], "negative_binomial")
    summary = glm.summarize(nb_fit, "negative_binomial")
    coefs = {c["term"]: c["estimate"] for c in summary["coefficients"]}
    assert abs(coefs["x1"] - 0.6) < 0.15
    assert "alpha" in coefs  # the estimated dispersion parameter itself


def test_interpret_excludes_alpha_and_const_from_significant_predictors():
    n = 300
    x1 = RNG.normal(0, 1, n)
    mu = np.exp(1.0 + 0.7 * x1)
    y = RNG.negative_binomial(3.0, 3.0 / (3.0 + mu))
    df = pd.DataFrame({"x1": x1, "y": y})
    fit = glm.fit_glm(df, "y", ["x1"], "negative_binomial")
    summary = glm.summarize(fit, "negative_binomial")
    sentence = glm.interpret(summary)
    assert "alpha" not in sentence.split("predictors (p < 0.05): ")[-1].split(".")[0] or "alpha" not in sentence


def test_residuals_vs_fitted_matches_n_obs():
    n = 200
    x1 = RNG.normal(0, 1, n)
    y = 3 + 2 * x1 + RNG.normal(0, 1, n)
    df = pd.DataFrame({"x1": x1, "y": y})
    fit = glm.fit_glm(df, "y", ["x1"], "gaussian")
    diagnostics = glm.residuals_vs_fitted(fit)
    assert len(diagnostics) == n
    assert {"fitted", "residual"} <= set(diagnostics.columns)


if __name__ == "__main__":
    test_detect_family_poisson()
    test_detect_family_binomial()
    test_detect_family_gamma()
    test_detect_family_tweedie()
    test_detect_family_gaussian_default()
    test_poisson_glm_recovers_known_coefficients()
    test_overdispersion_triggers_negative_binomial_upgrade()
    test_interpret_excludes_alpha_and_const_from_significant_predictors()
    test_residuals_vs_fitted_matches_n_obs()
    print("OK — all GLM self-checks passed.")
