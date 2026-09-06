"""Optional AI narrative layer. Not required for the app to work — every
caller must handle a None return (no API key, or the call failed) and fall
back to eda.rule_based_narrative() instead.

A full deepagents supervisor (ported from Project-Shonku's data_analyst
agent.py) is a Phase 2 concern once the Copilot chat page exists — a plain
single LLM call is all this narrative feature needs today.
"""

import logging
import os

logger = logging.getLogger(__name__)


def narrative_available() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def get_chat_model(temperature: float = 0.3):
    """A plain LangChain chat model, or None if no API key is set. Shared by
    the narrative helper below and the Copilot page's NL-to-SQL pipeline.
    """
    if not narrative_available():
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.environ.get("AGENTIC_DS_MODEL", "openai/gpt-4o-mini"),
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
    )


def generate_narrative(system_prompt: str, user_prompt: str) -> str | None:
    model = get_chat_model()
    if model is None:
        return None
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        return response.content
    except Exception:
        logger.exception("AI narrative generation failed")
        return None
