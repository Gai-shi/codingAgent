"""Model context-window lookup helpers."""

from __future__ import annotations


DEFAULT_CONTEXT_WINDOW = 128000

_MODEL_CONTEXT_WINDOWS = {
    "gpt-4.1": 1_047_576,
    "gpt-4.1-mini": 1_047_576,
    "gpt-4.1-nano": 1_047_576,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-5": 400_000,
    "gpt-5-mini": 400_000,
    "gpt-5-nano": 400_000,
}


def lookup_context_window(model_name: str) -> int | None:
    """Return the known context window for a model name, if configured locally."""
    normalized_name = model_name.strip().lower()
    return _MODEL_CONTEXT_WINDOWS.get(normalized_name)


def resolve_context_window(model_name: str, context_window_override: int | None = None) -> int:
    """Resolve context window from explicit override, known model registry, or fallback."""
    return context_window_override or lookup_context_window(model_name) or DEFAULT_CONTEXT_WINDOW
