from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_job.tools import GrepTool, ReadFileTool, create_default_tool_registry
from ai_job.tools.grep_tool import GREP_MAX_MATCHES, grep_text, normalize_file_type
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

    def test_read_file_text_rejects_invalid_missing_directory_and_non_utf8_inputs(self):
        self._write_text("dir/file.txt", "text\n")
        (self.workspace_root / "binary.bin").write_bytes(b"\xff")

        with self.assertRaisesRegex(ValueError, '"path" must be a string'):
            read_file_text({"path": None}, self.workspace_root)

        with self.assertRaisesRegex(FileNotFoundError, "file not found"):
            read_file_text({"path": "missing.txt"}, self.workspace_root)

        with self.assertRaisesRegex(ValueError, "not a file"):
            read_file_text({"path": "dir"}, self.workspace_root)

        with self.assertRaisesRegex(ValueError, "not valid UTF-8"):
            read_file_text({"path": "binary.bin"}, self.workspace_root)

    def test_grep_text_searches_utf8_files_with_type_filter_and_skips_protected(self):
        self._write_text("a.py", "needle in python\n")
        self._write_text("b.md", "needle in markdown\n")
        self._write_text(".git/config", "needle in git\n")
        self._write_text(".env", "needle in env\n")

        result = grep_text({"pattern": "needle", "type": "py"}, self.workspace_root)

        self.assertIn("a.py:1:needle in python", result)
        self.assertNotIn("b.md", result)
        self.assertNotIn(".git/config", result)
        self.assertNotIn(".env", result)

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

        with self.assertRaisesRegex(ValueError, '"type" must be a string'):
            normalize_file_type(123)

    def test_grep_text_rejects_invalid_arguments_and_regex(self):
        with self.assertRaisesRegex(ValueError, '"pattern" must be a non-empty string'):
            grep_text({"pattern": ""}, self.workspace_root)

        with self.assertRaisesRegex(ValueError, '"path" must be a string'):
            grep_text({"pattern": "needle", "path": None}, self.workspace_root)

        with self.assertRaisesRegex(ValueError, '"include_protected" must be a boolean'):
            grep_text({"pattern": "needle", "include_protected": "yes"}, self.workspace_root)

        with self.assertRaisesRegex(ValueError, "invalid regex pattern"):
            grep_text({"pattern": "["}, self.workspace_root)

    def test_grep_text_requires_callback_when_include_protected_is_true(self):
        self._write_text(".hidden/secret.txt", "needle\n")

        with self.assertRaisesRegex(PermissionError, "requires an approval callback"):
            grep_text(
                {"pattern": "needle", "path": ".hidden", "include_protected": True},
                self.workspace_root,
            )

    def test_grep_text_truncates_long_lines_and_max_matches(self):
        long_line = "needle " + ("x" * 400)
        self._write_text("long.txt", long_line + "\n")
        many_lines = "\n".join(f"needle {index}" for index in range(GREP_MAX_MATCHES + 5))
        self._write_text("many.txt", many_lines)

        long_result = grep_text({"pattern": "needle", "path": ".", "type": "txt"}, self.workspace_root)

        self.assertIn("long.txt:1:needle ", long_result)
        self.assertIn("...", long_result)
        self.assertIn(f"... truncated at {GREP_MAX_MATCHES} matches", long_result)

    def test_tool_classes_and_default_registry_wire_file_tools(self):
        registry = create_default_tool_registry(self.workspace_root)

        self.assertEqual(registry.names(), ["read_file", "grep"])
        self.assertIsInstance(registry.get("read_file"), ReadFileTool)
        self.assertIsInstance(registry.get("grep"), GrepTool)
