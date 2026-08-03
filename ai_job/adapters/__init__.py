"""Model API adapter exports."""

from .base import BaseToolCallAdapter
from .openai_tool_call_adapter import OpenAIToolCallAdapter

__all__ = [
    "BaseToolCallAdapter",
    "OpenAIToolCallAdapter",
]
