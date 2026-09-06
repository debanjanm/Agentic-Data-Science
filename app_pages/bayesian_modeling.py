import json

import numpy as np
import pandas as pd
import streamlit as st

from agentic_ds import bayesian as bys
from agentic_ds.ui_helpers import example_buttons, metric_row, narrative_button, target_predictor_family_picker

st.title("Bayesian modeling")
st.caption("Same target+predictors shape as Statistical modeling, but the output is a full posterior — a distribution of plausible coefficient values, not a single point estimate.")

# Same real, dataset-grounded questions as the GLM page — this page just
# answers them with a different backend (posterior distributions instead
# of point estimates + p-values).
EXAMPLES = [
    {"label": "Titanic — what predicts survival?", "dataset": "titanic", "target": "survived", "predictors": ["pclass", "sex", "age", "fare"]},
    {"label": "Titanic — what predicts family size aboard (siblings/spouses)?", "dataset": "titanic", "target": "sibsp", "predictors": ["pclass", "age", "fare"]},
    {"label": "Breast cancer — what predicts malignancy?", "dataset": "breast_cancer", "target": "target", "predictors": ["mean radius", "mean texture", "mean smoothness"]},
    {"label": "Diabetes — what predicts disease progression?", "dataset": "diabetes", "target": "target", "predictors": ["bmi", "bp", "s5"]},
]

example_buttons(
    EXAMPLES, "bayes_prefill", "bayes_result", "bayes_example",
    caption="Click one to load its dataset and pre-fill the model below (overwrites the current dataset).",
)

df = st.session_state.df
if df is None:
    st.warning("Upload or pick a dataset first (or click an example above).")
    st.page_link("app_pages/upload.py", label="Go to Upload & target", icon=":material/arrow_back:")
    st.stop()

st.caption(f"Source: {st.session_state.source_label} · {len(df):,} rows × {len(df.columns)} columns")

prefill = st.session_state.get("bayes_prefill") or {}

with st.container(border=True):
    target, predictors, family = target_predictor_family_picker(
        df, prefill, bys.suggest_family, bys.FAMILIES, bys.FAMILY_DESCRIPTIONS, default_n_predictors=3
    )

    with st.expander("Advanced: sampling settings"):
        st.caption("Higher values are slower but more reliable. Defaults are enough for a quick first look.")
        c1, c2, c3 = st.columns(3)
        draws = c1.number_input("Draws per chain", min_value=200, max_value=4000, value=500, step=100)
        tune = c2.number_input("Tuning steps", min_value=200, max_value=4000, value=500, step=100)
        chains = c3.number_input("Chains", min_value=2, max_value=4, value=2, step=1)

    if st.button("Fit model", icon=":material/play_arrow:"):
        try:
            with st.spinner("Sampling from the posterior — this can take several seconds..."):
                idata = bys.fit_bayesian(df, target, predictors, family, draws=draws, tune=tune, chains=chains)
                summary = bys.summarize(idata)
                ppc_samples = bys.posterior_predictive_samples(idata)
            st.session_state.bayes_result = {
                "summary": summary,
                "ppc_samples": ppc_samples.tolist(),
                "observed": df[target].dropna().tolist(),
            }
            st.session_state.bayes_prefill = None
        except Exception as e:
            st.error(f"Fit failed: {e}")

result = st.session_state.get("bayes_result")
if result:
    with st.container(border=True):
        st.markdown("**Result**")

        summary = result["summary"]
        if summary["converged"]:
            st.write(bys.interpret(summary))
        else:
            st.warning(bys.interpret(summary))

        metric_row({
            "max_rhat": summary["max_rhat"],
            "min_ess_bulk": summary["min_ess_bulk"],
            "converged": "yes" if summary["converged"] else "no",
        })

        st.caption(f"Coefficients (mean, sd, {int(summary['hdi_prob'] * 100)}% HDI)")
        st.dataframe(summary["coefficients"], width="stretch", hide_index=True)

        st.caption("Posterior predictive check — observed vs. simulated outcome distribution. They should look similar; if not, the model is misspecified.")
        observed = np.array(result["observed"])
        simulated = np.array(result["ppc_samples"])
        bins = np.histogram_bin_edges(np.concatenate([observed, simulated]), bins=25)
        obs_counts, _ = np.histogram(observed, bins=bins)
        sim_counts, _ = np.histogram(simulated, bins=bins)
        bin_centers = ((bins[:-1] + bins[1:]) / 2).round(2)
        ppc_df = pd.DataFrame({"observed": obs_counts, "simulated": sim_counts}, index=bin_centers)
        st.line_chart(ppc_df)

        narrative_button(
            "You are a statistics teacher. Given a fitted Bayesian model's posterior "
            "summary (coefficient means, HDIs, convergence diagnostics) as JSON, explain "
            "in 3-5 plain-English sentences: which predictors matter and in what "
            "direction, how confident to be given the interval widths, and any "
            "convergence caveats. No jargon, don't just restate the raw numbers.",
            json.dumps(summary, indent=2, default=str),
        )
