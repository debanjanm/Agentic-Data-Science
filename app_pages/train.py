import streamlit as st

from agentic_ds.modeling import CLASSIFICATION_ZOO, REGRESSION_ZOO, train_leaderboard

st.title("Train")

df = st.session_state.prepped_df if st.session_state.prepped_df is not None else st.session_state.df
target = st.session_state.target
task_type = st.session_state.task_type

if df is None or target is None:
    st.warning("Upload or pick a dataset and a target column first.")
    st.page_link("app_pages/upload.py", label="Go to Upload & target", icon=":material/arrow_back:")
    st.stop()

using_prepped = st.session_state.prepped_df is not None
st.caption(
    f"Source: {st.session_state.source_label}"
    + (" (cleaned)" if using_prepped else "")
    + f" · target: '{target}' · task: {task_type}"
)

zoo = CLASSIFICATION_ZOO if task_type == "classification" else REGRESSION_ZOO

with st.form("train_form"):
    model_names = st.multiselect("Models to try", list(zoo), default=list(zoo))
    test_size = st.slider("Holdout size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    submitted = st.form_submit_button("Run AutoML", icon=":material/play_arrow:")

if submitted:
    if not model_names:
        st.error("Pick at least one model.")
    else:
        with st.spinner(f"Training {len(model_names)} model(s)..."):
            result = train_leaderboard(df, target, task_type, model_names=model_names, test_size=test_size)
        st.session_state.train_result = result
        st.session_state.selected_model = result["leaderboard"].iloc[0]["model"]
        st.success("Done.")

result = st.session_state.train_result
if result is not None:
    with st.container(border=True):
        st.markdown("**Leaderboard preview**")
        st.dataframe(result["leaderboard"], width="stretch", hide_index=True)
        if st.button("View full leaderboard", icon=":material/arrow_forward:"):
            st.switch_page("app_pages/leaderboard.py")
