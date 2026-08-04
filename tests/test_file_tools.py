from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_job.tools.grep_tool import grep_text, normalize_file_type
from ai_job.tools.read_file_tool import read_file_text


class FileToolsTest(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self._tmp_dir.name).resolve()

    def tearDown(self):
        self._tmp_dir.cleanup()

    def _write_text(self, relative_path, text):
        path = self.workspace_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_read_file_text_reads_utf8_file_inside_workspace(self):
        self._write_text("src/app.py", "print('hello')\n")

        result = read_file_text({"path": "src/app.py"}, self.workspace_root)

        self.assertEqual(result, "print('hello')\n")

    def test_read_file_text_blocks_protected_paths_and_workspace_escape(self):
        self._write_text(".env", "OPENAI_API_KEY=secret\n")

        with self.assertRaisesRegex(PermissionError, "protected path"):
            read_file_text({"path": ".env"}, self.workspace_root)

        with self.assertRaisesRegex(ValueError, "escapes workspace"):
            read_file_text({"path": "../outside.txt"}, self.workspace_root)

    def test_grep_text_searches_utf8_files_with_type_filter_and_skips_protected(self):
        self._write_text("a.py", "needle in python\n")
        self._write_text("b.md", "needle in markdown\n")
        self._write_text(".git/config", "needle in git\n")

        result = grep_text({"pattern": "needle", "type": "py"}, self.workspace_root)

        self.assertIn("a.py:1:needle in python", result)
        self.assertNotIn("b.md", result)
        self.assertNotIn(".git/config", result)

    def test_grep_text_returns_no_matches_message(self):
        self._write_text("a.py", "haystack\n")

        result = grep_text({"pattern": "needle"}, self.workspace_root)

        self.assertEqual(result, "No matches.")

    def test_grep_text_requires_approval_for_hidden_or_protected_search_root(self):
        self._write_text(".hidden/secret.txt", "needle\n")
        approved_paths = []

        def approve(path):
            approved_paths.append(path)
            return True

        with self.assertRaisesRegex(PermissionError, "include_protected=true"):
            grep_text({"pattern": "needle", "path": ".hidden"}, self.workspace_root)

        result = grep_text(
            {"pattern": "needle", "path": ".hidden", "include_protected": True},
            self.workspace_root,
            approve,
        )

        self.assertEqual(approved_paths, [self.workspace_root / ".hidden"])
        self.assertIn(".hidden/secret.txt:1:needle", result)

    def test_grep_text_rejects_include_protected_when_user_denies(self):
        self._write_text(".hidden/secret.txt", "needle\n")

        with self.assertRaisesRegex(PermissionError, "user rejected"):
            grep_text(
                {"pattern": "needle", "path": ".hidden", "include_protected": True},
                self.workspace_root,
                lambda path: False,
            )

    def test_normalize_file_type_rejects_globs_and_paths(self):
        self.assertEqual(normalize_file_type(".py"), "py")
        self.assertIsNone(normalize_file_type(""))

        with self.assertRaisesRegex(ValueError, "simple file extension"):
            normalize_file_type("src/*.py")
