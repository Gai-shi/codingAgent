from __future__ import annotations

import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from ai_job.infra.http import HttpClientError, UrlLibHttpClient


class FakeHttpResponse:
    def __init__(self, body):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self._body


class UrlLibHttpClientTest(unittest.TestCase):
    def test_post_json_sends_json_request_and_returns_text_body(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return FakeHttpResponse("你好".encode("utf-8"))

        with patch("ai_job.infra.http.urllib_http_client.urllib.request.urlopen", fake_urlopen):
            result = UrlLibHttpClient().post_json(
                url="http://example.test/v1/chat/completions",
                payload={"model": "model", "messages": []},
                headers={"Authorization": "Bearer key", "Content-Type": "application/json"},
                timeout_seconds=2.5,
            )

        request = captured["request"]
        self.assertEqual(result, "你好")
        self.assertEqual(captured["timeout"], 2.5)
        self.assertEqual(request.full_url, "http://example.test/v1/chat/completions")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(json.loads(request.data.decode("utf-8")), {"model": "model", "messages": []})
        self.assertEqual(request.get_header("Authorization"), "Bearer key")
        self.assertEqual(request.get_header("Content-type"), "application/json")

    def test_post_json_wraps_http_error_with_response_body(self):
        http_error = urllib.error.HTTPError(
            url="http://example.test",
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=io.BytesIO("服务异常".encode("utf-8")),
        )

        with patch(
            "ai_job.infra.http.urllib_http_client.urllib.request.urlopen",
            side_effect=http_error,
        ):
            with self.assertRaisesRegex(HttpClientError, "HTTP 500：服务异常"):
                UrlLibHttpClient().post_json("http://example.test", {}, {}, 1.0)

    def test_post_json_wraps_url_error_reason(self):
        url_error = urllib.error.URLError("connection refused")

        with patch(
            "ai_job.infra.http.urllib_http_client.urllib.request.urlopen",
            side_effect=url_error,
        ):
            with self.assertRaisesRegex(HttpClientError, "connection refused"):
                UrlLibHttpClient().post_json("http://example.test", {}, {}, 1.0)
