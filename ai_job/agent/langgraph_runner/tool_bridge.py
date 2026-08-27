"""Bridge internal BaseTool instances into LangChain StructuredTool objects.

The wrapped tools are usable both for ``llm.bind_tools(...)`` (so the model sees
the same schema as the hand-written path) and for langgraph's prebuilt
``ToolNode`` (so the framework drives execution).

Execution goes through ``BaseTool.execute``, which validates arguments and
converts failures into ``Error: ...`` text. This keeps tool behavior identical
to the hand-written runner, so a hand-written vs framework A/B stays fair.
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from ...tools import BaseTool, ToolRegistry


def wrap_tool(tool: BaseTool) -> StructuredTool:
    """Wrap one internal BaseTool as a LangChain StructuredTool."""

    def _call(**arguments: object) -> str:
        return tool.execute(dict(arguments))

    return StructuredTool.from_function(
        func=_call,
        name=tool.name,
        description=tool.description,
        args_schema=tool.parameters_schema,
    )


def wrap_registry_tools(tool_registry: ToolRegistry) -> list[StructuredTool]:
    """Wrap every tool in the registry, preserving registration order."""
    return [wrap_tool(tool) for tool in tool_registry.tools()]
