"""Runtime composition for model-backed context compression."""

from __future__ import annotations

import json
from collections.abc import Callable

from ..agent.token_counting import estimate_message_tokens
from ..communication import (
    AssistantMessage,
    MessageHistory,
    SummaryMessage,
    UserMessage,
    message_history_to_debug_dicts,
)
from ..compress import CompressionManager, CompressionPlan
from ..infra.env import AppEnv
from ..provider_adapters import BaseChatModel, resolve_context_window
from ..tools import ToolRegistry


def create_compression_manager(app_env: AppEnv, chat_model: BaseChatModel) -> CompressionManager:
    context_window = resolve_context_window(
        app_env.openai_model,
        context_window_override=app_env.context_window_override,
    )
    return CompressionManager(
        context_window=context_window,
        reserve_tokens=app_env.compaction_reserve_tokens,
        keep_recent_tokens=app_env.compaction_keep_recent_tokens,
        token_counter=estimate_message_tokens,
        summarizer=create_summarizer(chat_model),
    )


def create_summarizer(chat_model: BaseChatModel) -> Callable[[CompressionPlan, MessageHistory], SummaryMessage]:
    empty_tool_registry = ToolRegistry([])

    def summarize(plan: CompressionPlan, history: MessageHistory) -> SummaryMessage:
        assistant_message = chat_model.complete(
            build_summary_messages(plan, history),
            empty_tool_registry,
        )
        return parse_summary_message(assistant_message)

    return summarize


def build_summary_messages(plan: CompressionPlan, history: MessageHistory) -> MessageHistory:
    content = "\n\n".join(
        [
            "请压缩以下对话上下文，输出 JSON，不要输出其它文本。",
            "JSON schema: {\"complete_turn_summary\": string, \"split_turn_summary\": string | null}",
            "complete_messages 是完整压缩掉的旧回合。",
            "split_messages 是被切开回合中未保留的前半段；如果没有该段，请返回 null。",
            "保留事实、用户目标、已做决定、工具调用结果、文件路径、错误信息和仍需继续的事项。",
            "不要编造未出现的信息。",
            "complete_messages:",
            json.dumps(
                message_history_to_debug_dicts(history[plan.complete_range.start : plan.complete_range.end]),
                ensure_ascii=False,
                indent=2,
            ),
            "split_messages:",
            json.dumps(
                message_history_to_debug_dicts(history[plan.split_range.start : plan.split_range.end])
                if plan.split_range is not None
                else [],
                ensure_ascii=False,
                indent=2,
            ),
        ]
    )
    return [UserMessage(content=content)]


def parse_summary_message(assistant_message: AssistantMessage) -> SummaryMessage:
    if not isinstance(assistant_message.content, str):
        raise RuntimeError("压缩摘要响应缺少文本 content")

    try:
        data = json.loads(assistant_message.content)
    except json.JSONDecodeError as exc:
        raise RuntimeError("压缩摘要响应不是合法 JSON") from exc

    if not isinstance(data, dict):
        raise RuntimeError("压缩摘要响应 JSON 必须是对象")

    complete_turn_summary = data.get("complete_turn_summary")
    split_turn_summary = data.get("split_turn_summary")
    if not isinstance(complete_turn_summary, str) or not complete_turn_summary.strip():
        raise RuntimeError("压缩摘要缺少 complete_turn_summary")
    if split_turn_summary is not None and not isinstance(split_turn_summary, str):
        raise RuntimeError("压缩摘要 split_turn_summary 必须是字符串或 null")

    return SummaryMessage(
        complete_turn_summary=complete_turn_summary,
        split_turn_summary=split_turn_summary,
    )
