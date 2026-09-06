import pandas as pd
import streamlit as st

from agentic_ds.data_io import BUILT_IN_DATASETS, BUILT_IN_TARGETS, load_builtin, load_csv
from agentic_ds.modeling import detect_task_type

st.title("Upload & target")
st.caption("Pick a dataset, then tell us what you want to predict.")


@st.cache_data
def _load_builtin_cached(name: str) -> pd.DataFrame:
    return load_builtin(name)


@st.cache_data
def _load_csv_cached(file) -> pd.DataFrame:
    return load_csv(file)


with st.container(border=True):
    source = st.segmented_control("Data source", ["Built-in dataset", "Upload CSV"], default="Built-in dataset")

    df = None
    source_label = None

    if source == "Built-in dataset":
        name = st.selectbox(
            "Dataset",
            list(BUILT_IN_DATASETS),
            format_func=lambda n: BUILT_IN_DATASETS[n],
        )
        try:
            df = _load_builtin_cached(name)
            source_label = f"built-in: {name}"
        except Exception as e:
            st.error(f"Could not load '{name}': {e}")
    else:
        uploaded = st.file_uploader("CSV file", type="csv")
        if uploaded is not None:
            try:
                df = _load_csv_cached(uploaded)
                source_label = f"upload: {uploaded.name}"
            except Exception as e:
                st.error(f"Could not read that CSV: {e}")

if df is not None:
    with st.container(horizontal=True):
        st.metric("Rows", f"{len(df):,}", border=True)
        st.metric("Columns", len(df.columns), border=True)
        st.metric("Missing values", f"{int(df.isna().sum().sum()):,}", border=True)

    with st.container(border=True):
        st.markdown("**Preview**")
        st.dataframe(df.head(20), width="stretch")

    with st.container(border=True):
        st.markdown("**Target**")
        default_target = BUILT_IN_TARGETS.get(name) if source == "Built-in dataset" else None
        columns = list(df.columns)
        default_index = columns.index(default_target) if default_target in columns else len(columns) - 1
        target = st.selectbox("Column to predict", columns, index=default_index)

        detected = detect_task_type(df, target)
        task_type = st.segmented_control(
            "Task type (auto-detected — override if wrong)",
            ["classification", "regression"],
            default=detected,
        )

    changed = source_label != st.session_state.source_label or target != st.session_state.target
    st.session_state.df = df
    st.session_state.source_label = source_label
    st.session_state.target = target
    st.session_state.task_type = task_type
    if changed:
        # A new dataset/target invalidates any prep/training done on the old one.
        st.session_state.prepped_df = None
        st.session_state.prep_plan = None
        st.session_state.train_result = None
        st.session_state.selected_model = None
        st.session_state.chat_history = []

    st.success(f"Ready — {task_type} on '{target}'.")
    if st.button("Continue to data profile", icon=":material/arrow_forward:"):
        st.switch_page("app_pages/profile.py")
else:
    st.info("Choose a built-in dataset or upload a CSV to get started.")
