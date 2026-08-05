from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from datetime import datetime
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

    def test_load_env_file_rejects_invalid_key_and_directory_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text("1INVALID=value\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "变量名非法"):
                load_env_file(env_path)

            with self.assertRaisesRegex(ValueError, "不是文件"):
                load_env_file(Path(tmp_dir))


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

    def test_load_from_current_environment_uses_defaults(self):
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "key", "OPENAI_MODEL": "model"},
            clear=True,
        ):
            app_env = EnvLoader.load_from_current_environment()

        self.assertEqual(app_env.openai_base_url, "https://api.openai.com/v1")
        self.assertEqual(app_env.timeout_seconds, 60.0)
        self.assertEqual(app_env.max_tool_rounds, 8)
        self.assertEqual(
            app_env.system_prompt,
            "You are a helpful coding agent. Use tools when you need workspace information.",
        )
        self.assertEqual(app_env.filter_terminal_log_level, "debug")

    def test_load_from_current_environment_rejects_invalid_timeout_and_rounds(self):
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "key", "OPENAI_MODEL": "model", "AI_JOB_TIMEOUT_SECONDS": "slow"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "AI_JOB_TIMEOUT_SECONDS 必须是数字"):
                EnvLoader.load_from_current_environment()

        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "key", "OPENAI_MODEL": "model", "AI_JOB_MAX_TOOL_ROUNDS": "many"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "AI_JOB_MAX_TOOL_ROUNDS 必须是整数"):
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
            daily_log_path = LogWrapper.log_path()
            self.assertEqual(daily_log_path.parent, log_path.parent)
            self.assertRegex(daily_log_path.name, r"^trace-\d{4}-\d{2}-\d{2}\.log$")
            log_text = daily_log_path.read_text(encoding="utf-8")
            self.assertIn(" INFO [trace] line1\\nline2\n", log_text)

    def test_log_wrapper_rejects_unknown_terminal_filter_level(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(ValueError, "FILTER_TERMINAL_LOG_LEVEL"):
                LogWrapper.configure(Path(tmp_dir) / "trace.log", "verbose")

    def test_log_wrapper_prints_only_messages_at_or_above_filter_level(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "trace.log"
            LogWrapper.configure(log_path=log_path, filter_terminal_log_level="warn")

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                LogWrapper.info("trace", "hidden")
                LogWrapper.warn("trace", "visible")
                LogWrapper.error("trace", "also visible")

            terminal_text = stderr.getvalue()
            self.assertNotIn("hidden", terminal_text)
            self.assertIn(" WARN [trace] visible\n", terminal_text)
            self.assertIn(" ERROR [trace] also visible\n", terminal_text)

            log_text = LogWrapper.log_path().read_text(encoding="utf-8")
            self.assertIn(" INFO [trace] hidden\n", log_text)

    def test_log_wrapper_rotates_daily_file_by_current_date(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_log_path = Path(tmp_dir) / "trace.log"
            LogWrapper.configure(log_path=base_log_path, filter_terminal_log_level="none")

            with patch.object(
                LogWrapper,
                "_now",
                side_effect=[
                    datetime(2026, 8, 5, 23, 59, 59),
                    datetime(2026, 8, 6, 0, 0, 1),
                ],
            ):
                LogWrapper.info("trace", "before midnight")
                LogWrapper.info("trace", "after midnight")

            first_log = Path(tmp_dir) / "trace-2026-08-05.log"
            second_log = Path(tmp_dir) / "trace-2026-08-06.log"
            self.assertIn("before midnight", first_log.read_text(encoding="utf-8"))
            self.assertIn("after midnight", second_log.read_text(encoding="utf-8"))

    def test_log_wrapper_cleanup_deletes_daily_logs_older_than_one_month(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_dir = Path(tmp_dir)
            base_log_path = log_dir / "trace.log"
            expired_log = log_dir / "trace-2026-07-04.log"
            cutoff_log = log_dir / "trace-2026-07-05.log"
            current_log = log_dir / "trace-2026-08-05.log"
            unrelated_log = log_dir / "other-2026-07-01.log"
            malformed_log = log_dir / "trace-not-a-date.log"
            for log_path in [expired_log, cutoff_log, current_log, unrelated_log, malformed_log]:
                log_path.write_text("log", encoding="utf-8")

            LogWrapper.configure(log_path=base_log_path, filter_terminal_log_level="none")

            with patch.object(LogWrapper, "_today", return_value=datetime(2026, 8, 5).date()):
                deleted_paths = LogWrapper.cleanup_expired_logs()

            self.assertEqual(deleted_paths, [expired_log])
            self.assertFalse(expired_log.exists())
            self.assertTrue(cutoff_log.exists())
            self.assertTrue(current_log.exists())
            self.assertTrue(unrelated_log.exists())
            self.assertTrue(malformed_log.exists())

    def test_log_wrapper_can_cleanup_expired_logs_in_background(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_dir = Path(tmp_dir)
            base_log_path = log_dir / "trace.log"
            expired_log = log_dir / "trace-2026-07-04.log"
            expired_log.write_text("log", encoding="utf-8")

            LogWrapper.configure(log_path=base_log_path, filter_terminal_log_level="none")

            with patch.object(LogWrapper, "_today", return_value=datetime(2026, 8, 5).date()):
                cleanup_thread = LogWrapper.cleanup_expired_logs_async()
                cleanup_thread.join(timeout=1)

            self.assertFalse(cleanup_thread.is_alive())
            self.assertFalse(expired_log.exists())
