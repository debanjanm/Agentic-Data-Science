"""Phase 9a: LLM-callable Tools wrapping the analysis functions already
built in Phases 1-8. No new analysis logic lives here — every tool below
is a thin function-calling wrapper around `eda.py`/`stats_tests.py`/
`glm.py`/`bayesian.py`/`timeseries.py`/`survival.py`/`sql_pipeline.py`.

Design constraint (see PLAN.md Phase 9a): tool arguments must be JSON-
serializable, so a tool can never take a DataFrame directly — the LLM
passes column *names* (strings), and the tool resolves them against the
currently-loaded dataset. That dataset is bound once, at construction
time, via `build_tools(df, ...)` returning a fresh list of closures — the
same "resolve server-side, never pass raw data through the LLM" shape
Shonku's original `tools.py` used for `dataset_name`/`csv_path`.

Every tool returns a JSON string (errors included, as `{"error": ...}`)
so a tool-calling loop can feed the result straight back as a
`ToolMessage` and — on error — let the model see what went wrong and
retry with different arguments, the same self-correction shape
`sql_pipeline.py` already uses for SQL.

This module only builds the tool list. The loop that actually calls a
model with these bound (`ChatOpenAI.bind_tools`) is Phase 9c, not here.
"""

import json

import pandas as pd
from langchain_core.tools import tool

from agentic_ds import bayesian as bys
from agentic_ds import glm
from agentic_ds import stats_tests as sx
from agentic_ds import survival as sv
from agentic_ds import timeseries as ts
from agentic_ds.eda import correlation_matrix, detect_outliers, numeric_summary, schema_overview, top_value_counts
from agentic_ds.sql_pipeline import nl_to_sql


def _safe_json(fn) -> str:
    """Run `fn`, JSON-encode the result; on failure return a `{"error": ...}`
    string instead of raising — lets a tool-calling loop see the failure
    and retry with different arguments rather than crashing the turn.
    """
    try:
        return json.dumps(fn(), indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


def build_tools(df: pd.DataFrame, source_label: str = "dataset", model=None, memory_db: str = "data/agentic_ds.db") -> list:
    """Build the full Tool list bound to `df`. Pass `model` (a LangChain
    chat model) to include the NL-to-SQL tool — omitted (None) if no LLM
    is configured, since that specific tool needs one internally
    (`smart_sql_query`'s chain-of-thought SQL generation), unlike every
    other tool here which is plain deterministic code.
    """

    # ── EDA ──────────────────────────────────────────────────────────

    @tool
    def get_schema_overview() -> str:
        """Column names, dtypes, missing-value counts, and cardinality for the current dataset."""
        return _safe_json(lambda: schema_overview(df).to_dict("records"))

    @tool
    def get_numeric_summary() -> str:
        """Descriptive statistics (count, mean, std, min, max, quartiles) for every numeric column."""
        return _safe_json(lambda: numeric_summary(df).reset_index().rename(columns={"index": "column"}).to_dict("records"))

    @tool
    def get_correlation_matrix() -> str:
        """Pearson correlation matrix between all numeric columns."""
        return _safe_json(lambda: correlation_matrix(df).round(3).to_dict())

    @tool
    def detect_column_outliers(column: str) -> str:
        """Detect outliers in a numeric column using the IQR method. Returns outlier count/pct and fences."""
        return _safe_json(lambda: detect_outliers(df, column))

    @tool
    def get_top_value_counts(column: str, limit: int = 10) -> str:
        """Most frequent values in a column (any dtype) and their counts."""
        return _safe_json(lambda: top_value_counts(df, column, limit).to_dict())

    # ── Hypothesis testing ──────────────────────────────────────────

    @tool
    def compare_two_groups(numeric_column: str, group_column: str) -> str:
        """Compare a numeric column across exactly 2 groups. Auto-picks Welch's t-test
        (both groups roughly normal) or Mann-Whitney U otherwise, and reports the result
        with an effect size and the assumption checks that decided which test ran."""
        return _safe_json(lambda: sx.compare_two_groups(df[numeric_column], df[group_column]))

    @tool
    def compare_many_groups(numeric_column: str, group_column: str) -> str:
        """Compare a numeric column across 3+ groups. Auto-picks one-way ANOVA (assumptions
        hold) or Kruskal-Wallis otherwise."""
        return _safe_json(lambda: sx.compare_many_groups(df[numeric_column], df[group_column]))

    @tool
    def compare_paired_samples(column_a: str, column_b: str) -> str:
        """Compare two numeric columns measured on the same subjects (e.g. before/after).
        Auto-picks a paired t-test or Wilcoxon signed-rank."""
        return _safe_json(lambda: sx.compare_paired(df[column_a], df[column_b]))

    @tool
    def test_correlation(column_a: str, column_b: str) -> str:
        """Test correlation between two numeric columns. Auto-picks Pearson (both roughly
        normal) or Spearman otherwise."""
        return _safe_json(lambda: sx.test_correlation(df[column_a], df[column_b]))

    @tool
    def test_categorical_association(column_a: str, column_b: str) -> str:
        """Test association between two categorical columns via a contingency table.
        Auto-picks chi-square independence, or Fisher's exact for small 2x2 tables."""
        return _safe_json(lambda: sx.test_categorical_association(df[column_a], df[column_b]))

    @tool
    def one_sample_test(column: str, hypothesized_value: float) -> str:
        """Test whether a numeric column's mean equals a hypothesized value. Auto-picks a
        one-sample t-test or Wilcoxon signed-rank."""
        return _safe_json(lambda: sx.one_sample_test(df[column], hypothesized_value))

    @tool
    def one_sample_proportion_test(count: int, nobs: int, hypothesized_value: float) -> str:
        """Z-test for whether an observed proportion (count out of nobs) equals a
        hypothesized value — e.g. is a conversion rate different from 5%."""
        return _safe_json(lambda: sx.one_sample_proportion_ztest(count, nobs, hypothesized_value))

    @tool
    def two_sample_proportion_test(count_a: int, nobs_a: int, count_b: int, nobs_b: int) -> str:
        """Z-test comparing two observed proportions — the core A/B test for a conversion-rate
        metric (e.g. control vs. treatment sign-up rate)."""
        return _safe_json(lambda: sx.two_sample_proportion_ztest(count_a, nobs_a, count_b, nobs_b))

    # ── GLM ──────────────────────────────────────────────────────────

    @tool
    def suggest_glm_family(target_column: str) -> str:
        """Suggest a GLM family (gaussian/binomial/poisson/gamma/tweedie) from the target
        column's shape, with the reason for the suggestion."""
        family, reason = glm.detect_family(df[target_column])
        return json.dumps({"family": family, "reason": reason})

    @tool
    def fit_glm_model(target_column: str, predictor_columns: list[str], family: str) -> str:
        """Fit a GLM (target ~ predictors) with the given family and return coefficients
        (estimate, std err, p-value, CI), fit statistics (AIC/BIC/deviance/pseudo-R²), and —
        for a poisson fit — an overdispersion check suggesting a negative-binomial upgrade.
        Call suggest_glm_family first if unsure which family to use."""

        def _run():
            fit_result = glm.fit_glm(df, target_column, predictor_columns, family)
            summary = glm.summarize(fit_result, family)
            if family == "poisson":
                summary["overdispersion"] = glm.check_overdispersion(fit_result)
            return summary

        return _safe_json(_run)

    # ── Bayesian ─────────────────────────────────────────────────────

    @tool
    def suggest_bayesian_family(target_column: str) -> str:
        """Suggest a Bayesian model family (linear/logistic/poisson) from the target
        column's shape, with the reason for the suggestion."""
        family, reason = bys.suggest_family(df[target_column])
        return json.dumps({"family": family, "reason": reason})

    @tool
    def fit_bayesian_model(target_column: str, predictor_columns: list[str], family: str) -> str:
        """Fit a Bayesian model (target ~ predictors) via MCMC and return posterior
        coefficient summaries (mean, sd, 94% HDI) and convergence diagnostics (r-hat, ESS).
        Slower than fit_glm_model (several seconds) — prefer GLM unless a full posterior
        distribution is actually needed. Call suggest_bayesian_family first if unsure."""

        def _run():
            idata = bys.fit_bayesian(df, target_column, predictor_columns, family)
            return bys.summarize(idata)

        return _safe_json(_run)

    # ── Time series ──────────────────────────────────────────────────

    @tool
    def check_series_stationarity(column: str) -> str:
        """Run ADF and KPSS stationarity tests on a numeric column (treated as an
        ordered series by row order). Reports "stationary"/"non-stationary"/"inconclusive"."""
        return _safe_json(lambda: ts.check_stationarity(df[column]))

    @tool
    def suggest_arima_differencing_order(column: str) -> str:
        """Suggest the ARIMA differencing order (d) needed to reach stationarity, with the
        stationarity check at each differencing step."""

        def _run():
            d, checks = ts.suggest_differencing_order(df[column])
            return {"suggested_d": d, "checks": checks}

        return _safe_json(_run)

    @tool
    def decompose_series(column: str, period: int) -> str:
        """Decompose a numeric column into trend/seasonal/residual components (additive
        model). `period` is the seasonal cycle length (e.g. 12 for monthly data with yearly
        seasonality)."""
        return _safe_json(lambda: ts.decompose(df[column], period).reset_index(drop=True).tail(20).to_dict("records"))

    @tool
    def compute_autocorrelation(column: str, nlags: int = 24) -> str:
        """ACF and PACF values for a numeric column, up to `nlags` lags — used to pick
        ARIMA's p/q orders (PACF cutoff suggests p, ACF cutoff suggests q)."""
        return _safe_json(lambda: ts.autocorrelation(df[column], nlags).to_dict("records"))

    @tool
    def test_series_whiteness(column: str, lags: int = 10) -> str:
        """Ljung-Box test: is there significant autocorrelation left in this column
        (raw series or model residuals), or does it look like white noise."""
        return _safe_json(lambda: ts.whiteness_test(df[column], lags))

    @tool
    def fit_arima_model(column: str, p: int, d: int, q: int, forecast_steps: int = 12) -> str:
        """Fit ARIMA(p,d,q) to a numeric column and return coefficients, fit stats
        (AIC/BIC), a residual whiteness check, and a forecast with 95% CIs. Call
        check_series_stationarity/suggest_arima_differencing_order and
        compute_autocorrelation first to pick sensible p/d/q."""
        return _safe_json(lambda: ts.fit_arima(df[column], (p, d, q), forecast_steps))

    # ── Survival analysis ───────────────────────────────────────────

    @tool
    def compute_kaplan_meier(duration_column: str, event_column: str) -> str:
        """Kaplan-Meier survival curve for the whole dataset: median survival time, event
        count, and the survival function over time."""
        return _safe_json(lambda: sv.kaplan_meier(df[duration_column], df[event_column]))

    @tool
    def compare_survival_groups(duration_column: str, event_column: str, group_column: str) -> str:
        """Compare survival between exactly 2 groups: a Kaplan-Meier curve per group plus
        the log-rank test for whether their survival genuinely differs."""
        return _safe_json(lambda: sv.compare_groups(df, duration_column, event_column, group_column))

    @tool
    def fit_cox_model(duration_column: str, event_column: str, covariate_columns: list[str]) -> str:
        """Fit a Cox proportional hazards model: hazard ratios with real inference
        (p-values, CIs) for each covariate, plus the concordance index (model discrimination,
        0.5 = chance, 1.0 = perfect)."""
        return _safe_json(lambda: sv.fit_cox(df, duration_column, event_column, covariate_columns))

    tools = [
        get_schema_overview,
        get_numeric_summary,
        get_correlation_matrix,
        detect_column_outliers,
        get_top_value_counts,
        compare_two_groups,
        compare_many_groups,
        compare_paired_samples,
        test_correlation,
        test_categorical_association,
        one_sample_test,
        one_sample_proportion_test,
        two_sample_proportion_test,
        suggest_glm_family,
        fit_glm_model,
        suggest_bayesian_family,
        fit_bayesian_model,
        check_series_stationarity,
        suggest_arima_differencing_order,
        decompose_series,
        compute_autocorrelation,
        test_series_whiteness,
        fit_arima_model,
        compute_kaplan_meier,
        compare_survival_groups,
        fit_cox_model,
    ]

    # ── SQL (needs a model — omitted if none configured) ────────────

    if model is not None:

        @tool
        def smart_sql_query(question: str) -> str:
            """Answer a natural-language question about the dataset by generating and
            running SQL (schema-aware, self-correcting on error). Use for aggregations,
            filtering, grouping, ranking — anything better expressed as a query than a
            statistical test."""
            result = nl_to_sql(question=question, df=df, dataset_label=source_label, model=model, memory_db=memory_db)
            return json.dumps(result, indent=2, default=str)

        tools.append(smart_sql_query)

    return tools
