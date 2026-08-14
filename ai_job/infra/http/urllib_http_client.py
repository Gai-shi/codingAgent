"""urllib-based HTTP client implementation."""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from typing import Any

from .base_http_client import BaseHttpClient, HttpClientError


class UrlLibHttpClient(BaseHttpClient):
    """HTTP client backed by Python's standard-library urllib."""

    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> str:
        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise HttpClientError(f"HTTP {exc.code}：{error_body}") from exc
        except urllib.error.URLError as exc:
            raise HttpClientError(str(exc.reason)) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise HttpClientError(f"request timed out after {timeout_seconds} seconds") from exc
