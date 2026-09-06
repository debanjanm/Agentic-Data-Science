import json

import streamlit as st

from agentic_ds import stats_tests as sx
from agentic_ds.ui_helpers import example_buttons, narrative_button

st.title("Hypothesis testing")
st.caption("Pick a question shape and columns — assumptions get checked automatically, and the matching test runs.")

# Realistic problem statements grounded in the app's own built-in datasets
# (see PLAN.md Phase 4) — each is a genuine, checkable question, not a
# placeholder. Clicking one loads its dataset and pre-fills the form below.
EXAMPLES = [
    {
        "label": "Titanic — does survival depend on sex?",
        "dataset": "titanic",
        "mode": "Categorical association",
        "col_a": "survived",
        "col_b": "sex",
    },
    {
        "label": "Titanic — did survivors pay higher fares?",
        "dataset": "titanic",
        "mode": "Compare two groups",
        "numeric_col": "fare",
        "group_col": "survived",
    },
    {
        "label": "Titanic — does passenger class affect fare?",
        "dataset": "titanic",
        "mode": "Compare 3+ groups",
        "numeric_col": "fare",
        "group_col": "pclass",
    },
    {
        "label": "Iris — does petal length differ by species?",
        "dataset": "iris",
        "mode": "Compare 3+ groups",
        "numeric_col": "petal length (cm)",
        "group_col": "target",
    },
    {
        "label": "Breast cancer — is mean radius associated with diagnosis?",
        "dataset": "breast_cancer",
        "mode": "Compare two groups",
        "numeric_col": "mean radius",
        "group_col": "target",
    },
    {
        "label": "Wine — is alcohol correlated with color intensity?",
        "dataset": "wine",
        "mode": "Correlation",
        "col_a": "alcohol",
        "col_b": "color_intensity",
    },
]

example_buttons(
    EXAMPLES, "stats_prefill", "stats_result", "example",
    caption="Click one to load its dataset and pre-fill the test below (overwrites the current dataset).",
)

df = st.session_state.df
if df is None:
    st.warning("Upload or pick a dataset first (or click an example above).")
    st.page_link("app_pages/upload.py", label="Go to Upload & target", icon=":material/arrow_back:")
    st.stop()

st.caption(f"Source: {st.session_state.source_label} · {len(df):,} rows × {len(df.columns)} columns")

prefill = st.session_state.stats_prefill or {}
MODES = [
    "Compare two groups",
    "Compare 3+ groups",
    "Paired samples",
    "Correlation",
    "Categorical association",
    "One-sample test",
]


def _idx(options: list, value, fallback: int = 0) -> int:
    return options.index(value) if value in options else fallback


numeric_cols = df.select_dtypes(include="number").columns.tolist()
all_cols = df.columns.tolist()

if not numeric_cols:
    st.error("This dataset has no numeric columns — most tests here need at least one.")
    st.stop()

result = None

with st.container(border=True):
    mode = st.segmented_control("What do you want to test?", MODES, default=MODES[_idx(MODES, prefill.get("mode"))])

    if mode == "Compare two groups":
        c1, c2 = st.columns(2)
        numeric_col = c1.selectbox("Numeric column", numeric_cols, index=_idx(numeric_cols, prefill.get("numeric_col")))
        group_col = c2.selectbox(
            "Group column (must have exactly 2 distinct values)", all_cols, index=_idx(all_cols, prefill.get("group_col"))
        )
        if st.button("Run test", icon=":material/play_arrow:"):
            n_groups = df[group_col].nunique()
            if n_groups != 2:
                st.error(f"'{group_col}' has {n_groups} distinct values — pick one with exactly 2, or use 'Compare 3+ groups'.")
            else:
                result = sx.compare_two_groups(df[numeric_col], df[group_col])

    elif mode == "Compare 3+ groups":
        c1, c2 = st.columns(2)
        numeric_col = c1.selectbox("Numeric column", numeric_cols, index=_idx(numeric_cols, prefill.get("numeric_col")))
        group_col = c2.selectbox(
            "Group column (must have 3+ distinct values)", all_cols, index=_idx(all_cols, prefill.get("group_col"))
        )
        if st.button("Run test", icon=":material/play_arrow:"):
            n_groups = df[group_col].nunique()
            if n_groups < 3:
                st.error(f"'{group_col}' has {n_groups} distinct values — pick one with 3+, or use 'Compare two groups'.")
            else:
                result = sx.compare_many_groups(df[numeric_col], df[group_col])

    elif mode == "Paired samples":
        st.caption("Two numeric columns measured on the *same* subjects (e.g. before/after).")
        c1, c2 = st.columns(2)
        col_a = c1.selectbox("Column A", numeric_cols, index=_idx(numeric_cols, prefill.get("col_a")))
        col_b = c2.selectbox("Column B", numeric_cols, index=_idx(numeric_cols, prefill.get("col_b"), fallback=min(1, len(numeric_cols) - 1)))
        if st.button("Run test", icon=":material/play_arrow:"):
            if col_a == col_b:
                st.error("Pick two different columns.")
            else:
                result = sx.compare_paired(df[col_a], df[col_b])

    elif mode == "Correlation":
        c1, c2 = st.columns(2)
        col_a = c1.selectbox("Column A", numeric_cols, index=_idx(numeric_cols, prefill.get("col_a")))
        col_b = c2.selectbox("Column B", numeric_cols, index=_idx(numeric_cols, prefill.get("col_b"), fallback=min(1, len(numeric_cols) - 1)))
        if st.button("Run test", icon=":material/play_arrow:"):
            if col_a == col_b:
                st.error("Pick two different columns.")
            else:
                result = sx.test_correlation(df[col_a], df[col_b])

    elif mode == "Categorical association":
        c1, c2 = st.columns(2)
        col_a = c1.selectbox("Column A", all_cols, index=_idx(all_cols, prefill.get("col_a")))
        col_b = c2.selectbox("Column B", all_cols, index=_idx(all_cols, prefill.get("col_b"), fallback=min(1, len(all_cols) - 1)))
        if st.button("Run test", icon=":material/play_arrow:"):
            if col_a == col_b:
                st.error("Pick two different columns.")
            else:
                result = sx.test_categorical_association(df[col_a], df[col_b])

    else:  # One-sample test
        numeric_col = st.selectbox("Numeric column", numeric_cols, index=_idx(numeric_cols, prefill.get("numeric_col")))
        default_mean = float(df[numeric_col].mean())
        popmean = st.number_input("Hypothesized value (H0: the mean equals this)", value=default_mean)
        if st.button("Run test", icon=":material/play_arrow:"):
            result = sx.one_sample_test(df[numeric_col], popmean)

if result is not None:
    st.session_state.stats_result = result
    st.session_state.stats_prefill = None

result = st.session_state.stats_result
if result:
    with st.container(border=True):
        st.markdown("**Result**")

        verdict = result.get("reject_null")
        interpretation = sx.interpret(result)
        if verdict is True:
            st.success(interpretation)
        elif verdict is False:
            st.info(interpretation)
        else:
            st.write(interpretation)

        with st.expander("Full result, including assumption checks"):
            st.json(result)

        narrative_button(
            "You are a statistics teacher. Given a hypothesis-test result as JSON, "
            "explain in 2-4 plain-English sentences: what was tested, what the result "
            "means practically (not just 'reject/fail to reject'), and how much "
            "confidence to place in it. No jargon, don't just restate the raw numbers.",
            json.dumps(result, indent=2, default=str),
        )
