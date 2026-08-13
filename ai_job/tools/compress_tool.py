"""compress_tool implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .base_tool import BaseTool, ToolExecutionContext


COMPRESS_TOOL_DESCRIPTION = (
    "Replace previous tool outputs in future model context when they contain lots of irrelevant "
    "content and only a small part is useful for the remaining task. "
    "Use this to keep the facts, file paths, line numbers, error messages, and reasoning evidence "
    "that matter while removing irrelevant bulk. "
    "Do not use this when the output still needs line-by-line analysis, when it is a stack trace, "
    "diff, patch, or test failure detail that may need exact quoting, or when you are unsure which "
    "details will matter later. "
    "Pass replacements as objects containing tool_name, the exact tool_arguments originally used, "
    "and replace_content. If the original tool call cannot be matched uniquely, no compression is "
    "applied."
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
                    "tool_name": {
                        "type": "string",
                        "description": "The name of the previous tool call to compress.",
                    },
                    "tool_arguments": {
                        "type": "object",
                        "description": "The exact JSON arguments of the previous tool call to compress.",
                    },
                    "replace_content": {
                        "type": "string",
                        "description": "The compressed content to show in future model context.",
                    },
                },
                "required": ["tool_name", "tool_arguments", "replace_content"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["replacements"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Replacement:
    tool_name: str
    tool_arguments: dict[str, Any]
    replace_content: str

    def match_key(self) -> tuple[str, str]:
        return (self.tool_name, _canonical_json(self.tool_arguments))


def compress_tool_messages(arguments: dict[str, Any], context: ToolExecutionContext) -> str:
    replacements = _parse_replacements(arguments)
    targets_by_key = _tool_message_targets_by_call_arguments(
        context.message_state.model_visible_history()
    )

    for replacement in replacements:
        matching_targets = targets_by_key.get(replacement.match_key(), [])
        if not matching_targets:
            raise ValueError(
                f"no previous tool result matches {replacement.tool_name} "
                f"with arguments {_canonical_json(replacement.tool_arguments)}"
            )
        if len(matching_targets) > 1:
            raise ValueError(
                f"multiple previous tool results match {replacement.tool_name} "
                f"with arguments {_canonical_json(replacement.tool_arguments)}"
            )

    for replacement in replacements:
        tool_message = targets_by_key[replacement.match_key()][0]
        tool_message.compressions.append(replacement.replace_content)

    return "Success"


def _parse_replacements(arguments: dict[str, Any]) -> list[Replacement]:
    raw_replacements = arguments.get("replacements")
    if not isinstance(raw_replacements, list) or not raw_replacements:
        raise ValueError('invalid arguments: "replacements" must be a non-empty list')

    replacements: list[Replacement] = []
    seen_match_keys: set[tuple[str, str]] = set()
    for index, raw_replacement in enumerate(raw_replacements):
        if not isinstance(raw_replacement, dict):
            raise ValueError(f'invalid arguments: "replacements[{index}]" must be an object')

        tool_name = raw_replacement.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError(
                f'invalid arguments: "replacements[{index}].tool_name" '
                "must be a non-empty string"
            )

        tool_arguments = raw_replacement.get("tool_arguments")
        if not isinstance(tool_arguments, dict):
            raise ValueError(
                f'invalid arguments: "replacements[{index}].tool_arguments" '
                "must be an object"
            )

        replace_content = raw_replacement.get("replace_content")
        if not isinstance(replace_content, str):
            raise ValueError(
                f'invalid arguments: "replacements[{index}].replace_content" '
                "must be a string"
            )

        replacement = Replacement(
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            replace_content=replace_content,
        )
        match_key = replacement.match_key()
        if match_key in seen_match_keys:
            raise ValueError(
                f"duplicate tool_name/tool_arguments in replacements: "
                f"{tool_name} {_canonical_json(tool_arguments)}"
            )
        seen_match_keys.add(match_key)
        replacements.append(replacement)

    return replacements


def _tool_message_targets_by_call_arguments(
    history: "MessageHistory",
) -> dict[tuple[str, str], list[Any]]:
    from ..communication import AssistantMessage, ToolMessage

    tool_calls_by_id: dict[str, tuple[str, dict[str, Any]]] = {}
    targets_by_key: dict[tuple[str, str], list[ToolMessage]] = {}
    for message in history:
        if isinstance(message, AssistantMessage):
            for tool_call in message.tool_calls:
                if tool_call.id in tool_calls_by_id:
                    raise ValueError(f'duplicate tool_call id in history: "{tool_call.id}"')
                tool_calls_by_id[tool_call.id] = (tool_call.name, tool_call.arguments)
            continue

        if not isinstance(message, ToolMessage):
            continue

        tool_call = tool_calls_by_id.get(message.tool_call_id)
        if tool_call is None:
            raise ValueError(
                f'missing assistant tool_call for ToolMessage.tool_call_id: "{message.tool_call_id}"'
            )
        tool_name, tool_arguments = tool_call
        match_key = (tool_name, _canonical_json(tool_arguments))
        targets_by_key.setdefault(match_key, []).append(message)

    return targets_by_key


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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
