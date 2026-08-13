"""compress_tool implementation."""

from __future__ import annotations

from typing import Any

from .base_tool import BaseTool, ToolExecutionContext


COMPRESS_TOOL_DESCRIPTION = (
    "Compress previous tool outputs for future model context. "
    "Pass replacements as objects containing tool_call_id and replace_content. "
    "If any tool_call_id is invalid, no compression is applied."
)
COMPRESS_TOOL_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "replacements": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "tool_call_id": {
                        "type": "string",
                        "description": "The tool_call_id of the ToolMessage to compress.",
                    },
                    "replace_content": {
                        "type": "string",
                        "description": "The compressed content to show in future model context.",
                    },
                },
                "required": ["tool_call_id", "replace_content"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["replacements"],
    "additionalProperties": False,
}


def compress_tool_messages(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    replacements = _parse_replacements(arguments)
    tool_messages = _tool_messages_by_id(context.message_history)

    missing_ids = [
        tool_call_id for tool_call_id, _ in replacements if tool_call_id not in tool_messages
    ]
    if missing_ids:
        raise ValueError(f'unknown tool_call_id: "{missing_ids[0]}"')

    for tool_call_id, replace_content in replacements:
        tool_messages[tool_call_id].compressions.append(replace_content)

    return "Success"


def _parse_replacements(arguments: dict[str, Any]) -> list[tuple[str, str]]:
    raw_replacements = arguments.get("replacements")
    if not isinstance(raw_replacements, list) or not raw_replacements:
        raise ValueError('invalid arguments: "replacements" must be a non-empty list')

    replacements: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    for index, raw_replacement in enumerate(raw_replacements):
        if not isinstance(raw_replacement, dict):
            raise ValueError(f'invalid arguments: "replacements[{index}]" must be an object')

        tool_call_id = raw_replacement.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise ValueError(
                f'invalid arguments: "replacements[{index}].tool_call_id" '
                "must be a non-empty string"
            )

        replace_content = raw_replacement.get("replace_content")
        if not isinstance(replace_content, str):
            raise ValueError(
                f'invalid arguments: "replacements[{index}].replace_content" '
                "must be a string"
            )

        if tool_call_id in seen_ids:
            raise ValueError(f'duplicate tool_call_id in replacements: "{tool_call_id}"')
        seen_ids.add(tool_call_id)
        replacements.append((tool_call_id, replace_content))

    return replacements


def _tool_messages_by_id(history: "MessageHistory") -> dict[str, Any]:
    from ..communication import ToolMessage

    tool_messages: dict[str, ToolMessage] = {}
    for message in history:
        if not isinstance(message, ToolMessage):
            continue
        if message.tool_call_id in tool_messages:
            raise ValueError(
                f'duplicate ToolMessage.tool_call_id in history: "{message.tool_call_id}"'
            )
        tool_messages[message.tool_call_id] = message

    return tool_messages


class CompressTool(BaseTool):
    name = "compress_tool"
    description = COMPRESS_TOOL_DESCRIPTION
    parameters_schema = COMPRESS_TOOL_PARAMETERS_SCHEMA

    def _run_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None,
    ) -> str:
        if context is None:
            raise ValueError("compress_tool requires tool execution context")
        return compress_tool_messages(arguments, context)
