"""Message visibility updates for agent-maintained context events."""

from __future__ import annotations

from ..communication import AssistantMessage, Message, MessageHistory, ToolMessage


COMPRESS_TOOL_NAME = "compress_tool"


class MessageVisibilityManager:
    """Adjust per-message model visibility after tool execution."""

    def apply_after_tool_execution(
        self,
        history: MessageHistory,
        *,
        assistant_message_index: int,
        tool_message_index: int,
        tool_name: str,
        success: bool,
    ) -> None:
        if tool_name != COMPRESS_TOOL_NAME:
            return

        self._validate_single_tool_execution_pair(
            history=history,
            assistant_message_index=assistant_message_index,
            tool_message_index=tool_message_index,
            tool_name=tool_name,
        )
        if success:
            self._hide_compress_tool_messages(history, through_index=tool_message_index)
            return

        self._set_visible(history, assistant_message_index, True)
        self._set_visible(history, tool_message_index, True)

    def _hide_compress_tool_messages(
        self,
        history: MessageHistory,
        *,
        through_index: int,
    ) -> None:
        for index, message in enumerate(history[: through_index + 1]):
            if not _is_single_compress_tool_assistant(message):
                continue

            self._set_visible(history, index, False)
            tool_call_id = message.tool_calls[0].id
            tool_message_index = _find_tool_message_index(
                history,
                tool_call_id=tool_call_id,
                start_index=index + 1,
                end_index=through_index,
            )
            if tool_message_index is not None:
                self._set_visible(history, tool_message_index, False)

    def _validate_single_tool_execution_pair(
        self,
        history: MessageHistory,
        *,
        assistant_message_index: int,
        tool_message_index: int,
        tool_name: str,
    ) -> None:
        assistant_message = _get_message(history, assistant_message_index)
        if not isinstance(assistant_message, AssistantMessage):
            raise ValueError("assistant_message_index must point to an AssistantMessage")
        if len(assistant_message.tool_calls) != 1:
            raise ValueError(
                "cannot adjust visibility for one tool_call inside a multi-tool AssistantMessage"
            )

        tool_call = assistant_message.tool_calls[0]
        if tool_call.name != tool_name:
            raise ValueError("assistant_message_index does not match tool_name")

        tool_message = _get_message(history, tool_message_index)
        if not isinstance(tool_message, ToolMessage):
            raise ValueError("tool_message_index must point to a ToolMessage")
        if tool_message.tool_call_id != tool_call.id:
            raise ValueError("tool_message_index does not match assistant tool_call id")

    def _set_visible(
        self,
        history: MessageHistory,
        index: int,
        visible_to_model: bool,
    ) -> None:
        _get_message(history, index).visible_to_model = visible_to_model


def _get_message(history: MessageHistory, index: int) -> Message:
    try:
        return history[index]
    except IndexError as exc:
        raise ValueError("message index out of range") from exc


def _is_single_compress_tool_assistant(message: Message) -> bool:
    return (
        isinstance(message, AssistantMessage)
        and len(message.tool_calls) == 1
        and message.tool_calls[0].name == COMPRESS_TOOL_NAME
    )


def _find_tool_message_index(
    history: MessageHistory,
    *,
    tool_call_id: str,
    start_index: int,
    end_index: int,
) -> int | None:
    for index in range(start_index, end_index + 1):
        message = history[index]
        if isinstance(message, ToolMessage) and message.tool_call_id == tool_call_id:
            return index
    return None
