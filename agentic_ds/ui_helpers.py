"""Small shared UI helpers so every Statistics page's fit-stats row, example
buttons, and "explain in plain English" block look the same instead of each
page hand-rolling its own copy (found via ponytail-audit — 7 near-identical
narrative blocks, 3 near-identical example-button grids, one duplicated
target/predictor/family picker)."""

import streamlit as st

from agentic_ds.data_io import load_builtin
from agentic_ds.llm import generate_narrative, narrative_available


def metric_row(items: dict, help_text: dict | None = None) -> None:
    """Render a flat dict of scalar stats as a row of bordered metric cards."""
    help_text = help_text or {}
    with st.container(horizontal=True):
        for key, value in items.items():
            label = key.replace("_", " ").capitalize()
            st.metric(label, value, border=True, help=help_text.get(key))


def narrative_button(
    system_prompt: str,
    user_prompt: str,
    *,
    label: str = "Explain in plain English",
    spinner_text: str = "Thinking...",
    fallback_caption: str = "Set OPENROUTER_API_KEY for a plain-English AI explanation (optional).",
    error_suffix: str = "",
) -> None:
    """The "optional AI narrative on top of a rule-based result" button —
    same shell everywhere: button + spinner + generate_narrative + markdown,
    or a caption if no API key is set. Callers only vary the prompts/wording.
    """
    if not narrative_available():
        st.caption(fallback_caption)
        return
    if st.button(label, icon=":material/auto_awesome:"):
        with st.spinner(spinner_text):
            explanation = generate_narrative(system_prompt, user_prompt)
        if explanation:
            st.markdown(explanation)
        else:
            st.error(f"{label} failed — check OPENROUTER_API_KEY.{error_suffix}")


def example_buttons(
    examples: list[dict],
    prefill_key: str,
    result_key: str,
    key_prefix: str,
    caption: str = "Click one to load its dataset and pre-fill the form below (overwrites the current dataset).",
) -> None:
    """The "Example questions" expander: each example is {"label", "dataset", ...}
    — clicking one loads that built-in dataset and stashes the example dict for
    the page to read as prefill defaults."""
    with st.expander("Example questions", icon=":material/lightbulb:"):
        st.caption(caption)
        with st.container(horizontal=True):
            for i, example in enumerate(examples):
                if st.button(example["label"], key=f"{key_prefix}_{i}"):
                    st.session_state.df = load_builtin(example["dataset"])
                    st.session_state.source_label = f"built-in: {example['dataset']}"
                    st.session_state[prefill_key] = example
                    st.session_state[result_key] = None
                    st.rerun()


def target_predictor_family_picker(
    df,
    prefill: dict,
    detect_family_fn,
    family_options: list[str],
    family_descriptions: dict,
    default_n_predictors: int = 5,
):
    """Target selectbox + predictor multiselect + auto-suggested/overridable
    family selectbox — the input shape shared by the GLM and Bayesian pages,
    which only differ in which family-detection function/list they use.
    Stops the page (via st.stop()) if no predictor is selected.
    """
    all_cols = df.columns.tolist()
    target_index = all_cols.index(prefill["target"]) if prefill.get("target") in all_cols else len(all_cols) - 1
    target = st.selectbox("Target (what you want to model)", all_cols, index=target_index)

    predictor_options = [c for c in all_cols if c != target]
    default_predictors = [p for p in prefill.get("predictors", []) if p in predictor_options]
    if not default_predictors:
        default_predictors = df[predictor_options].select_dtypes(include="number").columns.tolist()[:default_n_predictors]
    predictors = st.multiselect("Predictors", predictor_options, default=default_predictors)

    if not predictors:
        st.info("Pick at least one predictor.")
        st.stop()

    detected_family, reason = detect_family_fn(df[target])
    st.caption(f"Auto-suggested family: **{detected_family}** — {reason}")
    family = st.selectbox(
        "Family (override if you know better)",
        family_options,
        index=family_options.index(detected_family),
        format_func=lambda f: family_descriptions[f],
    )
    return target, predictors, family
