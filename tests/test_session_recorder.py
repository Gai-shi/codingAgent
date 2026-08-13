from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from ai_job.infra.session_recording import SessionRecorder


class SessionRecorderTest(unittest.TestCase):
    def test_session_recorder_writes_markdown_file_named_by_session_start_time(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir) / "sessions" / "sessions.md"
            started_at = datetime(2026, 8, 5, 10, 38, 12, 123000)

            with patch.object(SessionRecorder, "_now", return_value=datetime(2026, 8, 5, 10, 38, 13)):
                SessionRecorder.configure(
                    session_path=base_path,
                    session_started_at=started_at,
                    metadata={"workspace": "/tmp/workspace", "model": "model"},
                )
                SessionRecorder.record_session("UserMessage", "你好\n读取 README", "text")
                SessionRecorder.record_session("ToolCall read_file", {"path": "README.md"}, "json")

            session_path = Path(tmp_dir) / "sessions" / "sessions-20260805-103812-123.md"
            self.assertEqual(SessionRecorder.session_path(), session_path)
            session_text = session_path.read_text(encoding="utf-8")
            self.assertIn("# Session 20260805-103812-123", session_text)
            self.assertIn("- started_at: `2026-08-05T10:38:12.123`", session_text)
            self.assertIn("- workspace: `/tmp/workspace`", session_text)
            self.assertIn("## 10:38:13 UserMessage", session_text)
            self.assertIn("```text\n你好\n读取 README\n```", session_text)
            self.assertIn("## 10:38:13 ToolCall read_file", session_text)
            self.assertIn('```json\n{\n  "path": "README.md"\n}\n```', session_text)

    def test_session_recorder_uses_longer_fence_for_content_with_backticks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir) / "sessions" / "sessions.md"

            SessionRecorder.configure(session_path=base_path)
            SessionRecorder.record_session("ToolResult", "before\n```text\ninside\n```\nafter", "text")

            session_text = SessionRecorder.session_path().read_text(encoding="utf-8")
            self.assertIn("````text\nbefore\n```text\ninside\n```\nafter\n````", session_text)

    def test_session_recorder_rejects_unknown_record_format(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            SessionRecorder.configure(session_path=Path(tmp_dir) / "sessions.md")

            with self.assertRaisesRegex(ValueError, "未知会话记录格式"):
                SessionRecorder.record_session("UserMessage", "content", "xml")

    def test_session_recorder_cleanup_deletes_records_older_than_one_month_by_filename(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            session_dir = Path(tmp_dir) / "sessions"
            session_dir.mkdir()
            base_path = session_dir / "sessions.md"
            expired_record = session_dir / "sessions-20260705-103812-122.md"
            cutoff_record = session_dir / "sessions-20260705-103812-123.md"
            current_record = session_dir / "sessions-20260805-103812-123.md"
            unrelated_record = session_dir / "other-20260701-000000-000.md"
            malformed_record = session_dir / "sessions-not-a-date.md"
            for record_path in [
                expired_record,
                cutoff_record,
                current_record,
                unrelated_record,
                malformed_record,
            ]:
                record_path.write_text("record", encoding="utf-8")

            with patch.object(SessionRecorder, "_now", return_value=datetime(2026, 8, 5, 10, 38, 12, 123000)):
                SessionRecorder.configure(session_path=base_path)
                deleted_paths = SessionRecorder.cleanup_expired_session_records()

            self.assertEqual(deleted_paths, [expired_record])
            self.assertFalse(expired_record.exists())
            self.assertTrue(cutoff_record.exists())
            self.assertTrue(current_record.exists())
            self.assertTrue(unrelated_record.exists())
            self.assertTrue(malformed_record.exists())

    def test_session_recorder_can_cleanup_expired_records_in_background(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            session_dir = Path(tmp_dir) / "sessions"
            session_dir.mkdir()
            base_path = session_dir / "sessions.md"
            expired_record = session_dir / "sessions-20260705-103812-122.md"
            expired_record.write_text("record", encoding="utf-8")

            with patch.object(SessionRecorder, "_now", return_value=datetime(2026, 8, 5, 10, 38, 12, 123000)):
                SessionRecorder.configure(session_path=base_path)
                cleanup_thread = SessionRecorder.cleanup_expired_session_records_async()
                cleanup_thread.join(timeout=1)

            self.assertFalse(cleanup_thread.is_alive())
            self.assertFalse(expired_record.exists())


if __name__ == "__main__":
    unittest.main()
