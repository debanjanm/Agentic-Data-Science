import pickle

import pandas as pd
import streamlit as st

from agentic_ds.modeling import PARAM_GRIDS, feature_importance, score_pipeline, shap_importance, tune_model
from agentic_ds.ui_helpers import narrative_button

st.title("Leaderboard")

result = st.session_state.train_result
if result is None:
    st.warning("Train some models first.")
    st.page_link("app_pages/train.py", label="Go to Train", icon=":material/arrow_back:")
    st.stop()

leaderboard = result["leaderboard"]
task_type = result["task_type"]
primary_metric = result["primary_metric"]

best_row = leaderboard.iloc[0]
with st.container(horizontal=True):
    st.metric("Best model", best_row["model"], border=True)
    st.metric(f"Best {primary_metric}", best_row[primary_metric], border=True)
    st.metric("Models trained", len(leaderboard), border=True)

with st.container(border=True):
    st.dataframe(
        leaderboard,
        width="stretch",
        hide_index=True,
        column_config={primary_metric: st.column_config.ProgressColumn(primary_metric, min_value=0, max_value=1)}
        if primary_metric == "accuracy"
        else None,
    )
    st.bar_chart(leaderboard.set_index("model")[primary_metric])
    st.download_button(
        "Download leaderboard (.csv)",
        leaderboard.to_csv(index=False),
        file_name="leaderboard.csv",
        mime="text/csv",
        icon=":material/download:",
    )

model_name = st.selectbox("Inspect a model", leaderboard["model"].tolist(), index=0)
pipeline = result["pipelines"][model_name]
X_test, y_test = result["X_test"], result["y_test"]

col_a, col_b = st.columns(2)

with col_a, st.container(border=True, height="stretch"):
    st.markdown("**Feature importance**")
    st.caption("Permutation importance — how much shuffling a feature hurts the score.")
    with st.spinner("Computing..."):
        importances = feature_importance(pipeline, X_test, y_test, task_type)
    st.bar_chart(importances.set_index("feature")["importance"])

    with st.expander("Cross-check with SHAP"):
        st.caption(
            "A different methodology (mean |SHAP value|, computed on a small sample — "
            "it calls the model many times per row). Useful as a sanity check if the two "
            "disagree on which features matter most."
        )
        if st.button("Compute SHAP", icon=":material/functions:", key=f"shap_{model_name}"):
            with st.spinner("Computing SHAP values..."):
                shap_values = shap_importance(pipeline, X_test, task_type)
            st.bar_chart(shap_values.set_index("feature")["mean_abs_shap"])

with col_b, st.container(border=True, height="stretch"):
    if task_type == "classification":
        st.markdown("**Confusion matrix**")
        y_pred = pipeline.predict(X_test)
        labels = sorted(y_test.unique())
        matrix = pd.crosstab(
            pd.Series(y_test, name="actual"), pd.Series(y_pred, name="predicted")
        ).reindex(index=labels, columns=labels, fill_value=0)
        st.dataframe(matrix, width="stretch")
    else:
        st.markdown("**Predicted vs actual**")
        y_pred = pipeline.predict(X_test)
        chart_df = pd.DataFrame({"actual": y_test.values, "predicted": y_pred})
        st.scatter_chart(chart_df, x="actual", y="predicted")

with st.container(border=True):
    if model_name in PARAM_GRIDS:
        if st.button(f"Tune {model_name} (RandomizedSearchCV)", icon=":material/tune:"):
            with st.spinner("Searching hyperparameters..."):
                tuned = tune_model(model_name, pipeline, result["X_train"], result["y_train"], task_type)
            new_metrics = score_pipeline(tuned, X_test, y_test, task_type)
            result["pipelines"][model_name] = tuned
            row = result["leaderboard"]["model"] == model_name
            for metric, value in new_metrics.items():
                result["leaderboard"].loc[row, metric] = value
            result["leaderboard"] = result["leaderboard"].sort_values(primary_metric, ascending=False).reset_index(drop=True)
            st.success(f"Tuned. New {primary_metric}: {new_metrics[primary_metric]}")
            st.rerun()
    else:
        st.caption(f"'{model_name}' has no tunable hyperparameters in this app.")

    with st.container(horizontal=True):
        if st.button("Use this model for prediction", icon=":material/arrow_forward:"):
            st.session_state.selected_model = model_name
            st.switch_page("app_pages/predict.py")
        st.download_button(
            "Download trained pipeline (.pkl)",
            pickle.dumps(pipeline),
            file_name=f"{model_name}.pkl",
            mime="application/octet-stream",
            icon=":material/download:",
            help="A pickled scikit-learn Pipeline (preprocessing + model). Only unpickle files you trust.",
        )

with st.container(border=True):
    st.markdown("**Executive summary**")
    narrative_button(
        "You are writing an executive summary of an AutoML run for a non-technical "
        "stakeholder. Cover: which model won overall and how well it performs, the top "
        "2-3 features driving predictions for the inspected model, and one or two "
        "caveats (small holdout size, class imbalance, etc. if relevant). 4-6 sentences, "
        "plain language, no jargon.",
        f"TASK: {task_type} predicting '{result['target']}'\n\n"
        f"LEADERBOARD (best first):\n{leaderboard.to_string(index=False)}\n\n"
        f"TOP FEATURES FOR INSPECTED MODEL ({model_name}):\n{importances.head(5).to_string(index=False)}",
        label="Generate summary",
        spinner_text="Writing summary...",
        fallback_caption="Set OPENROUTER_API_KEY to enable an AI-written executive summary (optional).",
    )
