import json

import pandas as pd
import streamlit as st
from lifelines.datasets import load_rossi

from agentic_ds import survival as sv
from agentic_ds.ui_helpers import metric_row, narrative_button

st.title("Survival analysis")
st.caption("Kaplan-Meier survival curves, the log-rank test, and Cox proportional hazards regression.")

with st.container(border=True):
    if st.button("Load the Rossi recidivism study (real data)", icon=":material/lightbulb:"):
        st.session_state.df = load_rossi()
        st.session_state.source_label = "built-in: rossi (lifelines) — 432-subject recidivism study"
        st.session_state.surv_result = None
        st.session_state.surv_cox_result = None
        st.rerun()
    st.caption(
        "432 recently released prisoners followed for 52 weeks: `week`/`arrest` are duration/event, "
        "`fin` (received financial aid), `age`, `race`, `wexp` (work experience), `mar` (married), "
        "`paro` (on parole), `prio` (prior convictions) are covariates. Bundled with `lifelines` itself "
        "— real published data, not synthetic."
    )

df = st.session_state.df
if df is None:
    st.warning("Load the example dataset above (this page needs duration + event columns most datasets don't have).")
    st.stop()

all_cols = df.columns.tolist()
st.caption(f"Source: {st.session_state.source_label} · {len(df):,} rows × {len(df.columns)} columns")

with st.container(border=True):
    c1, c2 = st.columns(2)
    duration_col = c1.selectbox("Duration column", all_cols, index=all_cols.index("week") if "week" in all_cols else 0)
    event_col = c2.selectbox(
        "Event column (1 = event observed, 0 = censored)", all_cols, index=all_cols.index("arrest") if "arrest" in all_cols else min(1, len(all_cols) - 1)
    )

with st.container(border=True):
    st.markdown("**Kaplan-Meier**")
    group_options = ["(none — one overall curve)"] + [c for c in all_cols if c not in (duration_col, event_col)]
    group_col = st.selectbox("Compare groups by (optional, needs exactly 2 groups)", group_options)

    if st.button("Compute survival curve", icon=":material/play_arrow:"):
        try:
            if group_col == "(none — one overall curve)":
                result = {"mode": "single", "km": sv.kaplan_meier(df[duration_col], df[event_col])}
            else:
                result = {"mode": "compare", "compare": sv.compare_groups(df, duration_col, event_col, group_col)}
            st.session_state.surv_result = result
        except Exception as e:
            st.error(f"Failed: {e}")

    result = st.session_state.get("surv_result")
    if result:
        if result["mode"] == "single":
            km = result["km"]
            st.write(sv.interpret_km(km))
            curve = {row["time"]: row["survival"] for row in km["curve"]}
            st.line_chart(curve)
        else:
            compare = result["compare"]
            verdict_text = sv.interpret_logrank(compare["logrank"])
            if compare["logrank"]["reject_null"]:
                st.success(verdict_text)
            else:
                st.write(verdict_text)
            chart_data = {}
            for label, km in compare["curves"].items():
                st.caption(sv.interpret_km(km))
                for row in km["curve"]:
                    chart_data.setdefault(row["time"], {})[label] = row["survival"]

            chart_df = pd.DataFrame.from_dict(chart_data, orient="index").sort_index().ffill()
            st.line_chart(chart_df)
        with st.expander("Full result"):
            st.json(result)

with st.container(border=True):
    st.markdown("**Cox proportional hazards**")
    covariate_options = [c for c in all_cols if c not in (duration_col, event_col)]
    default_covariates = df[covariate_options].select_dtypes(include="number").columns.tolist()
    covariates = st.multiselect("Covariates", covariate_options, default=default_covariates)

    if covariates and st.button("Fit Cox model", icon=":material/play_arrow:"):
        try:
            st.session_state.surv_cox_result = sv.fit_cox(df, duration_col, event_col, covariates)
        except Exception as e:
            st.error(f"Fit failed: {e}")

    cox_result = st.session_state.get("surv_cox_result")
    if cox_result:
        st.write(sv.interpret_cox(cox_result))
        metric_row({k: v for k, v in cox_result.items() if k != "coefficients"})

        st.caption("Coefficients (hazard ratio > 1 = higher hazard, < 1 = lower/protective)")
        st.dataframe(cox_result["coefficients"], width="stretch", hide_index=True)

        narrative_button(
            "You are a statistics teacher. Given a fitted Cox proportional hazards model's "
            "summary (hazard ratios, p-values, concordance index) as JSON, explain in 3-5 "
            "plain-English sentences: which covariates matter and whether they raise or lower "
            "risk, how good the model's discrimination is, and any caveats. No jargon.",
            json.dumps(cox_result, indent=2, default=str),
        )
