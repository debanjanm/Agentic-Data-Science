import pandas as pd
import streamlit as st

from agentic_ds.modeling import predict

st.title("Predict")

result = st.session_state.train_result
if result is None or st.session_state.selected_model is None:
    st.warning("Train and pick a model on the Leaderboard page first.")
    st.page_link("app_pages/leaderboard.py", label="Go to Leaderboard", icon=":material/arrow_back:")
    st.stop()

model_name = st.session_state.selected_model
pipeline = result["pipelines"][model_name]
task_type = result["task_type"]
feature_df = result["X_train"]

with st.container(horizontal=True):
    st.metric("Model", model_name, border=True)
    st.metric("Task", task_type, border=True)

mode = st.segmented_control("Input", ["Upload CSV", "Single row"], default="Upload CSV")

if mode == "Upload CSV":
    with st.container(border=True):
        st.caption(f"CSV must contain these columns: {', '.join(feature_df.columns)}")
        uploaded = st.file_uploader("New data", type="csv", key="predict_upload")
        if uploaded is not None:
            new_df = pd.read_csv(uploaded)
            missing = set(feature_df.columns) - set(new_df.columns)
            if missing:
                st.error(f"Missing required column(s): {', '.join(missing)}")
            else:
                predictions = predict(pipeline, new_df[feature_df.columns], task_type)
                st.dataframe(predictions, width="stretch")
                st.download_button(
                    "Download predictions",
                    predictions.to_csv(index=False),
                    file_name="predictions.csv",
                    mime="text/csv",
                    icon=":material/download:",
                )
else:
    with st.form("single_row_form"):
        values = {}
        for column in feature_df.columns:
            if pd.api.types.is_numeric_dtype(feature_df[column]):
                values[column] = st.number_input(column, value=float(feature_df[column].median()))
            else:
                options = sorted(feature_df[column].dropna().unique().tolist())
                values[column] = st.selectbox(column, options)
        submitted = st.form_submit_button("Predict", icon=":material/play_arrow:")

    if submitted:
        row_df = pd.DataFrame([values])[feature_df.columns]
        predictions = predict(pipeline, row_df, task_type)
        row = predictions.iloc[0]
        with st.container(horizontal=True):
            st.metric("Prediction", row["prediction"], border=True)
            if "confidence" in predictions.columns:
                st.metric("Confidence", f"{row['confidence']:.1%}", border=True)
        with st.expander("Input used"):
            st.dataframe(row_df, width="stretch", hide_index=True)
