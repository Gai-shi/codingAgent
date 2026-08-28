"""LangGraph-based runner implementation (parallel to the hand-written AgentRunner).

This subpackage is only imported when the langgraph extra is installed and the
langgraph runner is explicitly selected, so the core package stays framework-free.
"""

__all__ = [
    "LangGraphAgentRunner",
    "internal_history_to_langchain_messages",
    "internal_message_to_langchain_message",
    "langchain_ai_message_to_internal_assistant_message",
]


def __getattr__(name: str):
    if name == "LangGraphAgentRunner":
        from .runner import LangGraphAgentRunner

        return LangGraphAgentRunner

    if name in {
        "internal_history_to_langchain_messages",
        "internal_message_to_langchain_message",
        "langchain_ai_message_to_internal_assistant_message",
    }:
        from . import message_conversion

        return getattr(message_conversion, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
