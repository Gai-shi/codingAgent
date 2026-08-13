"""Message visibility updates for agent-maintained context events."""

from __future__ import annotations

from ..communication import AssistantMessage, Message, MessageHistory, MessageState, ToolMessage


COMPRESS_TOOL_NAME = "compress_tool"


class MessageVisibilityManager:
    """Adjust per-message model visibility after tool execution."""

    def apply_after_tool_batch(
        self,
        *,
        message_state: MessageState,
        assistant_message_index: int,
        tool_message_indexes: list[int],
    ) -> None:
        history = message_state.history
        assistant_message = _get_message(history, assistant_message_index)
        if not isinstance(assistant_message, AssistantMessage):
            raise ValueError("assistant_message_index must point to an AssistantMessage")

        if not _is_compress_tool_assistant(assistant_message):
            return

        self._validate_compress_tool_batch(
            history=history,
            assistant_message=assistant_message,
            tool_message_indexes=tool_message_indexes,
        )
        if self._all_tool_messages_succeeded(history, tool_message_indexes):
            self._hide_compress_tool_messages(
                message_state,
                through_index=tool_message_indexes[-1],
            )
            return

        self._set_visible(history, assistant_message_index, True)
        for tool_message_index in tool_message_indexes:
            self._set_visible(history, tool_message_index, True)

    def _hide_compress_tool_messages(
        self,
        message_state: MessageState,
        *,
        through_index: int,
    ) -> None:
        history = message_state.history
        pending_tool_call_ids: set[str] = set()
        for index in range(message_state.context_start_index, through_index + 1):
            message = history[index]

            if _is_compress_tool_assistant(message):
                self._set_visible(history, index, False)
                pending_tool_call_ids.update(tool_call.id for tool_call in message.tool_calls)
                continue

            if isinstance(message, ToolMessage) and message.tool_call_id in pending_tool_call_ids:
                self._set_visible(history, index, False)
                pending_tool_call_ids.remove(message.tool_call_id)

    def _validate_compress_tool_batch(
        self,
        history: MessageHistory,
        *,
        assistant_message: AssistantMessage,
        tool_message_indexes: list[int],
    ) -> None:
        if len(tool_message_indexes) != len(assistant_message.tool_calls):
            raise ValueError("tool_message_indexes must match assistant tool_calls")

        for tool_call, tool_message_index in zip(
            assistant_message.tool_calls,
            tool_message_indexes,
        ):
            tool_message = _get_message(history, tool_message_index)
            if not isinstance(tool_message, ToolMessage):
                raise ValueError("tool_message_indexes must point to ToolMessage instances")
            if tool_message.tool_call_id != tool_call.id:
                raise ValueError("tool_message index does not match assistant tool_call id")

    def _all_tool_messages_succeeded(
        self,
        history: MessageHistory,
        tool_message_indexes: list[int],
    ) -> bool:
        return all(
            isinstance(_get_message(history, tool_message_index), ToolMessage)
            and _get_message(history, tool_message_index).content == "Success"
            for tool_message_index in tool_message_indexes
        )

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


def _is_compress_tool_assistant(message: Message) -> bool:
    return (
        isinstance(message, AssistantMessage)
        and bool(message.tool_calls)
        and all(tool_call.name == COMPRESS_TOOL_NAME for tool_call in message.tool_calls)
    )
