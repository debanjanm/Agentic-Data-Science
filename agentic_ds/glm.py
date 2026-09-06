"""GLM statistical modeling: family auto-suggestion, fitting via
statsmodels, coefficient tables with real inference (p-values, CIs),
and residual diagnostics.

statsmodels only — see PLAN.md "Modules and libraries for Phases 4-6" for
why this beats stretching sklearn's PoissonRegressor/GammaRegressor/
TweedieRegressor into this role (they fit coefficients but give no
p-values, CIs, or deviance — inference output is the point of this page).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

OVERDISPERSION_THRESHOLD = 1.5

FAMILY_DESCRIPTIONS = {
    "gaussian": "Gaussian (identity link) — continuous, roughly symmetric target",
    "binomial": "Binomial (logit link) — binary (0/1) or proportion target",
    "poisson": "Poisson (log link) — non-negative integer count target",
    "negative_binomial": "Negative binomial (log link) — overdispersed counts (variance ≫ mean)",
    "gamma": "Gamma (log link) — positive, right-skewed continuous target",
    "tweedie": "Tweedie (log link) — non-negative with a point mass at zero plus a continuous positive part",
    "inverse_gaussian": "Inverse Gaussian (identity link) — positive continuous, heavier-tailed alternative to Gamma",
}


def _build_family(name: str):
    if name == "gaussian":
        return sm.families.Gaussian()
    if name == "binomial":
        return sm.families.Binomial()
    if name == "poisson":
        return sm.families.Poisson()
    if name == "gamma":
        return sm.families.Gamma(link=sm.families.links.Log())
    if name == "tweedie":
        return sm.families.Tweedie(var_power=1.5)
    if name == "inverse_gaussian":
        return sm.families.InverseGaussian()
    raise ValueError(f"Unknown family '{name}'. Use fit_glm's negative_binomial path instead of building it here.")


def detect_family(y: pd.Series) -> tuple[str, str]:
    """Suggest a family from the target's shape. Order matters: binary
    check first, then integer counts, then zero-inflated, then skewed
    positive, falling back to Gaussian. Negative binomial and inverse
    Gaussian are never auto-suggested — NB only makes sense as a Poisson
    upgrade *after* fitting shows overdispersion (see `check_overdispersion`),
    and inverse Gaussian is a manual alternative to Gamma, not a distinct
    shape to detect.
    """
    y = pd.Series(y).dropna()
    unique_vals = set(y.unique())

    if unique_vals <= {0, 1}:
        return "binomial", "target is binary (0/1)"

    is_integerish = bool(np.allclose(y, y.round()))
    if is_integerish and (y >= 0).all():
        return "poisson", "target looks like non-negative integer counts"

    zero_fraction = float((y == 0).mean())
    if (y >= 0).all() and zero_fraction > 0.05 and (y > 0).any():
        return "tweedie", f"target is non-negative with {zero_fraction:.0%} exact zeros plus a continuous positive part"

    if (y > 0).all():
        skew = float(y.skew())
        if skew > 1:
            return "gamma", f"target is positive and right-skewed (skew={skew:.2f})"

    return "gaussian", "target is continuous with no strong distributional signal"


def check_overdispersion(fit_result) -> dict:
    """Pearson chi2 / residual df — a ratio well above 1 means Poisson's
    equal mean-variance assumption doesn't hold and negative binomial
    (which adds a free dispersion parameter) is the better fit.
    """
    dispersion = float(fit_result.pearson_chi2 / fit_result.df_resid)
    return {"dispersion": round(dispersion, 4), "overdispersed": dispersion > OVERDISPERSION_THRESHOLD}


def _design_matrix(df: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    X = pd.get_dummies(df[predictors], drop_first=True)
    return sm.add_constant(X, has_constant="add")


def fit_glm(df: pd.DataFrame, target: str, predictors: list[str], family_name: str):
    """Fit a GLM (or, for negative_binomial, statsmodels' MLE-based discrete
    NegativeBinomial model — GLM's own NegativeBinomial family needs a
    pre-specified fixed dispersion, which defeats the point of using NB in
    the first place). Returns (fit_result, family_name_actually_used).
    """
    data = df[[target] + predictors].dropna()
    X = _design_matrix(data, predictors)
    y = data[target]

    if family_name == "negative_binomial":
        fit_result = sm.NegativeBinomial(y, X).fit(disp=0)
    else:
        fit_result = sm.GLM(y, X, family=_build_family(family_name)).fit()

    return fit_result


def summarize(fit_result, family_name: str) -> dict:
    """Coefficient table + fit statistics, in a plain-dict shape the UI and
    the narrative layer can both consume without touching statsmodels
    objects directly.
    """
    conf_int = fit_result.conf_int()
    conf_int.columns = ["ci_lower", "ci_upper"]
    coef_table = pd.DataFrame({
        "term": fit_result.params.index,
        "estimate": fit_result.params.values.round(4),
        "std_err": fit_result.bse.values.round(4),
        "statistic": fit_result.tvalues.values.round(4),
        "p_value": fit_result.pvalues.values.round(6),
        "ci_lower": conf_int["ci_lower"].values.round(4),
        "ci_upper": conf_int["ci_upper"].values.round(4),
    })

    result = {
        "family": family_name,
        "n_obs": int(fit_result.nobs),
        "coefficients": coef_table.to_dict("records"),
        "aic": round(float(fit_result.aic), 2),
        "bic": round(float(fit_result.bic), 2) if not np.isnan(fit_result.bic) else None,
        "log_likelihood": round(float(fit_result.llf), 2),
    }

    if hasattr(fit_result, "deviance"):
        result["deviance"] = round(float(fit_result.deviance), 4)
    try:
        result["pseudo_r2_mcfadden"] = round(float(1 - fit_result.llf / fit_result.llnull), 4)
    except Exception:
        pass

    return result


def residuals_vs_fitted(fit_result) -> pd.DataFrame:
    """Deviance residuals where available (the standard GLM diagnostic),
    Pearson residuals as a fallback for model types that don't expose them.
    """
    fitted = fit_result.predict()
    if hasattr(fit_result, "resid_deviance"):
        residuals = fit_result.resid_deviance
    elif hasattr(fit_result, "resid_pearson"):
        residuals = fit_result.resid_pearson
    else:
        residuals = fit_result.resid
    return pd.DataFrame({"fitted": np.asarray(fitted), "residual": np.asarray(residuals)})


def interpret(summary: dict, overdispersion: dict | None = None) -> str:
    """Rule-based, always-available floor — no LLM required."""
    non_predictor_terms = {"const", "alpha"}  # alpha is negative binomial's dispersion param, not a predictor
    sig_terms = [
        c["term"] for c in summary["coefficients"] if c["p_value"] < 0.05 and c["term"] not in non_predictor_terms
    ]
    sentence = f"{summary['family']} GLM, {summary['n_obs']} observations, AIC = {summary['aic']}."
    if sig_terms:
        sentence += f" Statistically significant predictors (p < 0.05): {', '.join(sig_terms)}."
    else:
        sentence += " No predictor reached statistical significance at p < 0.05."
    if "pseudo_r2_mcfadden" in summary:
        sentence += f" McFadden's pseudo-R² = {summary['pseudo_r2_mcfadden']}."
    if overdispersion and overdispersion["overdispersed"]:
        sentence += (
            f" Note: dispersion = {overdispersion['dispersion']} (> {OVERDISPERSION_THRESHOLD}) — "
            "Poisson's equal mean-variance assumption looks violated; negative binomial is a better fit."
        )
    return sentence
