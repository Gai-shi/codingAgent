"""Base HTTP client contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class HttpClientError(RuntimeError):
    """Raised when an HTTP client cannot complete a request."""


class BaseHttpClient(ABC):
    """Minimal JSON-over-HTTP client contract used by provider adapters."""

    @abstractmethod
    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> str:
        """POST a JSON payload and return the response body as text."""
