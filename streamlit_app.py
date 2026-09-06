"""Agentic Data Science — tabular data workbench. Entry point.

Pages live in app_pages/ and share state through st.session_state. Two
problem types share the "Data" pages (df, source_label) then diverge:
AutoML (target, task_type, train_result, ...) and Statistics (its own
column pickers — see PLAN.md's "Architecture change for Phase 4+"). See
PLAN.md for the full product design.
"""

import streamlit as st

st.set_page_config(page_title="Agentic data science", page_icon=":material/analytics:", layout="wide")

for key, default in {
    "df": None,
    "source_label": None,
    "target": None,
    "task_type": None,
    "prepped_df": None,
    "prep_plan": None,
    "train_result": None,
    "selected_model": None,
    "chat_history": [],
    "stats_prefill": None,
    "stats_result": None,
    "glm_prefill": None,
    "glm_result": None,
    "bayes_prefill": None,
    "bayes_result": None,
    "ts_series": None,
    "ts_source_label": None,
    "ts_arima_result": None,
    "surv_result": None,
    "surv_cox_result": None,
}.items():
    st.session_state.setdefault(key, default)

page = st.navigation({
    "Data": [
        st.Page("app_pages/upload.py", title="Upload & target", icon=":material/upload_file:"),
        st.Page("app_pages/profile.py", title="Data profile", icon=":material/query_stats:"),
        st.Page("app_pages/data_prep.py", title="Data prep", icon=":material/auto_fix_high:"),
    ],
    "AutoML": [
        st.Page("app_pages/train.py", title="Train", icon=":material/model_training:"),
        st.Page("app_pages/leaderboard.py", title="Leaderboard", icon=":material/leaderboard:"),
        st.Page("app_pages/predict.py", title="Predict", icon=":material/insights:"),
    ],
    "Statistics": [
        st.Page("app_pages/hypothesis_testing.py", title="Hypothesis testing", icon=":material/rule:"),
        st.Page("app_pages/statistical_modeling.py", title="Statistical modeling", icon=":material/functions:"),
        st.Page("app_pages/bayesian_modeling.py", title="Bayesian modeling", icon=":material/scatter_plot:"),
        st.Page("app_pages/timeseries.py", title="Time series", icon=":material/show_chart:"),
        st.Page("app_pages/survival_analysis.py", title="Survival analysis", icon=":material/monitor_heart:"),
    ],
    "Copilot": [
        st.Page("app_pages/copilot.py", title="Copilot", icon=":material/forum:"),
    ],
})

page.run()
