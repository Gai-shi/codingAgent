from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from ai_job.infra.env import EnvLoader, load_env_file
from ai_job.infra.logging import LogWrapper


class EnvFileLoaderTest(unittest.TestCase):
    def test_load_env_file_supports_export_quotes_and_keeps_shell_values(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "# ignored",
                        "export OPENAI_API_KEY='from-file'",
                        'OPENAI_MODEL="model-from-file"',
                        "KEEP_EXISTING=from-file",
                        "BARE=value",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"KEEP_EXISTING": "from-shell"}, clear=True):
                load_env_file(env_path)

                self.assertEqual(os.environ["OPENAI_API_KEY"], "from-file")
                self.assertEqual(os.environ["OPENAI_MODEL"], "model-from-file")
                self.assertEqual(os.environ["KEEP_EXISTING"], "from-shell")
                self.assertEqual(os.environ["BARE"], "value")

    def test_load_env_file_rejects_malformed_line(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text("NOT_A_PAIR\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "缺少 ="):
                load_env_file(env_path)


class EnvLoaderTest(unittest.TestCase):
    def test_load_from_current_environment_returns_typed_app_env(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "key",
                "OPENAI_MODEL": "model",
                "OPENAI_BASE_URL": "http://localhost:8787/v1/",
                "AI_JOB_TIMEOUT_SECONDS": "2.5",
                "AI_JOB_MAX_TOOL_ROUNDS": "3",
                "AI_JOB_SYSTEM_PROMPT": "system",
                "FILTER_TERMINAL_LOG_LEVEL": "warn",
            },
            clear=True,
        ):
            app_env = EnvLoader.load_from_current_environment()

        self.assertEqual(app_env.openai_api_key, "key")
        self.assertEqual(app_env.openai_model, "model")
        self.assertEqual(app_env.openai_base_url, "http://localhost:8787/v1")
        self.assertEqual(app_env.timeout_seconds, 2.5)
        self.assertEqual(app_env.max_tool_rounds, 3)
        self.assertEqual(app_env.system_prompt, "system")
        self.assertEqual(app_env.filter_terminal_log_level, "warn")

    def test_load_from_current_environment_requires_model_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY, OPENAI_MODEL"):
                EnvLoader.load_from_current_environment()

    def test_load_from_current_environment_rejects_non_positive_rounds(self):
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "key", "OPENAI_MODEL": "model", "AI_JOB_MAX_TOOL_ROUNDS": "0"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "AI_JOB_MAX_TOOL_ROUNDS 必须大于 0"):
                EnvLoader.load_from_current_environment()


class LogWrapperTest(unittest.TestCase):
    def test_log_wrapper_writes_single_line_log_and_respects_terminal_filter(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "nested" / "trace.log"
            LogWrapper.configure(log_path=log_path, filter_terminal_log_level="none")

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                LogWrapper.info("trace", "line1\nline2")

            self.assertEqual(stderr.getvalue(), "")
            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn(" INFO [trace] line1\\nline2\n", log_text)

    def test_log_wrapper_rejects_unknown_terminal_filter_level(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(ValueError, "FILTER_TERMINAL_LOG_LEVEL"):
                LogWrapper.configure(Path(tmp_dir) / "trace.log", "verbose")
