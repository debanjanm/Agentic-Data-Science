import pandas as pd
import streamlit as st

from agentic_ds.data_prep import apply_plan, propose_plan

st.title("Data prep")
st.caption("Propose a cleaning plan, review it, then apply only what you approve.")

df = st.session_state.df
target = st.session_state.target
if df is None or target is None:
    st.warning("Upload or pick a dataset and a target column first.")
    st.page_link("app_pages/upload.py", label="Go to Upload & target", icon=":material/arrow_back:")
    st.stop()

if st.session_state.prepped_df is not None:
    with st.container(border=True):
        st.info(f"A cleaned dataset is active: {len(st.session_state.prepped_df):,} rows × {len(st.session_state.prepped_df.columns)} columns. Train will use it.")
        if st.button("Discard and start over from the raw data", icon=":material/undo:"):
            st.session_state.prepped_df = None
            st.session_state.prep_plan = None
            st.rerun()

if st.button("Propose cleaning plan", icon=":material/auto_fix_high:"):
    st.session_state.prep_plan = propose_plan(df, target)

plan = st.session_state.prep_plan
if plan is None:
    st.stop()

if not plan:
    st.success("Nothing to clean — no missing values, duplicates, high-cardinality columns, or heavy outliers found.")
    st.stop()

with st.container(border=True):
    st.markdown("**Proposed steps**")
    st.caption("Uncheck anything you don't want, then apply.")
    plan_df = pd.DataFrame(plan)
    edited = st.data_editor(
        plan_df,
        width="stretch",
        hide_index=True,
        disabled=["column", "issue", "action"],
        column_config={"approved": st.column_config.CheckboxColumn("apply?")},
        key="prep_plan_editor",
    )

    if st.button("Apply approved steps", icon=":material/play_arrow:"):
        steps = edited.to_dict("records")
        cleaned = apply_plan(df, steps)
        st.session_state.prepped_df = cleaned
        st.session_state.prep_plan = steps
        st.session_state.train_result = None  # a changed dataset invalidates any prior training run
        st.success(f"Applied {sum(1 for s in steps if s['approved'])} step(s).")
        st.rerun()

if st.session_state.prepped_df is not None:
    st.markdown("**Before / after**")
    col_a, col_b = st.columns(2)
    with col_a, st.container(border=True, height="stretch"):
        st.caption("Raw")
        st.dataframe(df.head(10), width="stretch")
        st.caption(f"{len(df):,} rows × {len(df.columns)} columns, {int(df.isna().sum().sum())} missing values")
    with col_b, st.container(border=True, height="stretch"):
        st.caption("Cleaned")
        st.dataframe(st.session_state.prepped_df.head(10), width="stretch")
        cleaned = st.session_state.prepped_df
        st.caption(f"{len(cleaned):,} rows × {len(cleaned.columns)} columns, {int(cleaned.isna().sum().sum())} missing values")
