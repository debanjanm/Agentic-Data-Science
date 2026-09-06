import numpy as np
import pandas as pd
import streamlit as st

from agentic_ds.eda import correlation_matrix, detect_outliers, numeric_summary, rule_based_narrative, schema_overview, top_correlated_pairs, top_value_counts
from agentic_ds.ui_helpers import metric_row, narrative_button

st.title("Data profile")

df = st.session_state.df
if df is None:
    st.warning("Upload or pick a dataset first.")
    st.page_link("app_pages/upload.py", label="Go to Upload & target", icon=":material/arrow_back:")
    st.stop()

st.caption(f"Source: {st.session_state.source_label} · target: '{st.session_state.target}' · task: {st.session_state.task_type}")

with st.container(border=True):
    st.markdown("**Summary**")
    for bullet in rule_based_narrative(df):
        st.markdown(f"- {bullet}")

    narrative_button(
        "You are a data analyst. Given a dataset schema and summary statistics, "
        "write a concise (4-6 bullet points) narrative of what stands out: data "
        "quality issues, interesting distributions, and anything worth investigating "
        "before modeling. Plain markdown bullets, no preamble.",
        f"SCHEMA:\n{schema_overview(df).to_string(index=False)}\n\nNUMERIC SUMMARY:\n{numeric_summary(df).to_string()}",
        label="Generate AI summary",
        fallback_caption="Set OPENROUTER_API_KEY to enable an AI-generated summary (optional).",
        error_suffix=" Showing the rule-based summary above instead.",
    )

with st.container(border=True):
    st.markdown("**Schema**")
    st.dataframe(schema_overview(df), width="stretch", hide_index=True)

with st.container(border=True):
    st.markdown("**Numeric summary**")
    summary = numeric_summary(df)
    if summary.empty:
        st.caption("No numeric columns.")
    else:
        st.dataframe(summary, width="stretch")

col_a, col_b = st.columns(2)

with col_a, st.container(border=True, height="stretch"):
    st.markdown("**Correlations**")
    corr = correlation_matrix(df)
    if corr.empty:
        st.caption("Need at least 2 numeric columns.")
    else:
        st.dataframe(corr, width="stretch")
        st.caption("Top correlated pairs")
        st.dataframe(top_correlated_pairs(corr), width="stretch", hide_index=True)

with col_b, st.container(border=True, height="stretch"):
    st.markdown("**Column inspector**")
    column = st.selectbox("Column", df.columns)
    if np.issubdtype(df[column].dtype, np.number):
        counts, bin_edges = np.histogram(df[column].dropna(), bins=20)
        labels = [f"{bin_edges[i]:.2f}" for i in range(len(counts))]
        st.bar_chart(pd.Series(counts, index=labels, name="count"))
        st.caption("Outlier check (IQR method)")
        outliers = detect_outliers(df, column)
        if "error" in outliers:
            st.caption(outliers["error"])
        else:
            metric_row({
                "outlier_count": outliers["outlier_count"],
                "outlier_pct": f"{outliers['outlier_pct']}%",
                "lower_fence": outliers["lower_fence"],
                "upper_fence": outliers["upper_fence"],
            })
    else:
        st.bar_chart(top_value_counts(df, column))
