import json

import streamlit as st

from agentic_ds import glm
from agentic_ds.ui_helpers import example_buttons, metric_row, narrative_button, target_predictor_family_picker

st.title("Statistical modeling (GLM)")
st.caption("Pick a target and predictors — the family is auto-suggested from the target's shape, with real inference (p-values, CIs) on the output, not just predictions.")

# Realistic problem statements grounded in the app's own built-in datasets.
# Each preset only chooses target+predictors — the family shown is always
# whatever detect_family() actually decides, never asserted up front, so
# an example can't silently claim a family the real detection wouldn't pick.
EXAMPLES = [
    {"label": "Titanic — what predicts survival?", "dataset": "titanic", "target": "survived", "predictors": ["pclass", "sex", "age", "fare"]},
    {"label": "Titanic — what predicts family size aboard (siblings/spouses)?", "dataset": "titanic", "target": "sibsp", "predictors": ["pclass", "age", "fare"]},
    {"label": "Breast cancer — what predicts malignancy?", "dataset": "breast_cancer", "target": "target", "predictors": ["mean radius", "mean texture", "mean smoothness"]},
    {"label": "Breast cancer — what predicts area-measurement error?", "dataset": "breast_cancer", "target": "area error", "predictors": ["mean radius", "mean texture", "mean area"]},
    {"label": "Diabetes — what predicts disease progression?", "dataset": "diabetes", "target": "target", "predictors": ["bmi", "bp", "s5"]},
]

example_buttons(
    EXAMPLES, "glm_prefill", "glm_result", "glm_example",
    caption="Click one to load its dataset and pre-fill the model below (overwrites the current dataset).",
)

df = st.session_state.df
if df is None:
    st.warning("Upload or pick a dataset first (or click an example above).")
    st.page_link("app_pages/upload.py", label="Go to Upload & target", icon=":material/arrow_back:")
    st.stop()

st.caption(f"Source: {st.session_state.source_label} · {len(df):,} rows × {len(df.columns)} columns")

prefill = st.session_state.get("glm_prefill") or {}

with st.container(border=True):
    target, predictors, family = target_predictor_family_picker(
        df, prefill, glm.detect_family, list(glm.FAMILY_DESCRIPTIONS), glm.FAMILY_DESCRIPTIONS, default_n_predictors=5
    )

    if st.button("Fit model", icon=":material/play_arrow:"):
        try:
            fit_result = glm.fit_glm(df, target, predictors, family)
            st.session_state.glm_result = {
                "summary": glm.summarize(fit_result, family),
                "diagnostics": glm.residuals_vs_fitted(fit_result).to_dict("list"),
                "overdispersion": glm.check_overdispersion(fit_result) if family == "poisson" else None,
            }
            st.session_state.glm_prefill = None
        except Exception as e:
            st.error(f"Fit failed: {e}")

result = st.session_state.get("glm_result")
if result:
    with st.container(border=True):
        st.markdown("**Result**")

        summary = result["summary"]
        overdispersion = result["overdispersion"]
        if overdispersion and overdispersion["overdispersed"]:
            st.warning(glm.interpret(summary, overdispersion))
            if st.button("Refit as negative binomial", icon=":material/autorenew:"):
                fit_result = glm.fit_glm(df, target, predictors, "negative_binomial")
                st.session_state.glm_result = {
                    "summary": glm.summarize(fit_result, "negative_binomial"),
                    "diagnostics": glm.residuals_vs_fitted(fit_result).to_dict("list"),
                    "overdispersion": None,
                }
                st.rerun()
        else:
            st.write(glm.interpret(summary, overdispersion))

        fit_stats = {k: v for k, v in summary.items() if k != "coefficients"}
        metric_row(fit_stats)

        st.caption("Coefficients")
        st.dataframe(summary["coefficients"], width="stretch", hide_index=True)

        st.caption("Residuals vs. fitted (deviance residuals where available) — look for patterns, which would suggest a misspecified family or missing predictor")
        st.scatter_chart(result["diagnostics"], x="fitted", y="residual")

        narrative_button(
            "You are a statistics teacher. Given a fitted GLM's summary (coefficients, "
            "p-values, fit statistics) as JSON, explain in 3-5 plain-English sentences: "
            "which predictors matter and in what direction, how well the model fits, and "
            "any caveats (e.g. an overdispersion warning, a weak pseudo-R²). No jargon, "
            "don't just restate the raw numbers.",
            json.dumps(summary, indent=2, default=str),
        )
