"""Minimal LangGraph checkpoint wiring demo.

This module intentionally stays outside the CLI runtime. It shows where a
LangGraph checkpointer plugs in without replacing ai_job's SessionRecorder.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph


class CheckpointDemoState(TypedDict):
    counter: int


def build_checkpoint_demo_graph():
    """Build a tiny graph compiled with an in-memory checkpointer."""
    graph = StateGraph(CheckpointDemoState)
    graph.add_node("increment", _increment)
    graph.set_entry_point("increment")
    graph.add_edge("increment", END)
    return graph.compile(checkpointer=InMemorySaver())


def _increment(state: CheckpointDemoState) -> dict[str, int]:
    return {"counter": state["counter"] + 1}
