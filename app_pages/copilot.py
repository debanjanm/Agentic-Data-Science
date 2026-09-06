import pandas as pd
import streamlit as st

from agentic_ds.llm import get_chat_model, narrative_available
from agentic_ds.sql_pipeline import nl_to_sql

MEMORY_DB = "data/agentic_ds.db"

st.title("Copilot")
st.caption("Ask a question about the current dataset in plain English — it's converted to SQL and run for you.")

df = st.session_state.df
if df is None:
    st.warning("Upload or pick a dataset first.")
    st.page_link("app_pages/upload.py", label="Go to Upload & target", icon=":material/arrow_back:")
    st.stop()

if not narrative_available():
    st.info("Set OPENROUTER_API_KEY to use the Copilot (see .env.example). Unlike the Data profile "
             "page's summary, there's no rule-based fallback for open-ended questions.")
    st.stop()

dataset_label = st.session_state.source_label or "dataset"
st.caption(f"Dataset: {dataset_label} · {len(df):,} rows × {len(df.columns)} columns")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("data") is not None and not message["data"].empty:
            st.dataframe(message["data"], width="stretch", hide_index=True)

question = st.chat_input("e.g. \"average of each numeric column by category\" or \"top 5 rows by X\"")
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = nl_to_sql(
                question=question,
                df=df,
                dataset_label=dataset_label,
                model=get_chat_model(),
                memory_db=MEMORY_DB,
            )

        if result["status"] == "success":
            reply = f"```sql\n{result['sql']}\n```\n{result['rows_returned']} row(s) returned."
            st.markdown(reply)
            data = pd.DataFrame(result["data"])
            st.dataframe(data, width="stretch", hide_index=True)
            st.session_state.chat_history.append({"role": "assistant", "content": reply, "data": data})
        else:
            reply = f"Couldn't answer that after {result['attempts']} attempt(s). Last error: {result['error']}"
            st.error(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply, "data": None})
