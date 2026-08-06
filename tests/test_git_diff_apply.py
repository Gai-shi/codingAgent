from __future__ import annotations

import unittest

from ai_job.tools.git_diff_apply import GitDiffApplyError, apply_file_patch_to_text
from ai_job.tools.git_diff_parser import parse_git_diff


def first_file(patch: str):
    return parse_git_diff(patch).files[0]


class GitDiffApplyTest(unittest.TestCase):
    def test_apply_modify_patch_by_unique_old_content(self):
        patch = """diff --git a/src/calc.py b/src/calc.py
--- a/src/calc.py
+++ b/src/calc.py
@@ -10,2 +10,2 @@
 def add(a, b):
-    return a - b
+    return a + b
"""

        result = apply_file_patch_to_text(
            first_file(patch),
            "# header\n\ndef add(a, b):\n    return a - b\n",
        )

        self.assertEqual(result.operation, "modify")
        self.assertEqual(result.path, "src/calc.py")
        self.assertEqual(result.content, "# header\n\ndef add(a, b):\n    return a + b\n")

    def test_apply_modify_patch_supports_no_newline_marker(self):
        patch = """diff --git a/a.txt b/a.txt
--- a/a.txt
+++ b/a.txt
@@ -1 +1 @@
-old
\\ No newline at end of file
+new
\\ No newline at end of file
"""

        result = apply_file_patch_to_text(first_file(patch), "old")

        self.assertEqual(result.content, "new")

    def test_apply_modify_patch_rejects_missing_and_ambiguous_matches(self):
        missing_patch = """diff --git a/a.txt b/a.txt
--- a/a.txt
+++ b/a.txt
@@ -1 +1 @@
-old
+new
"""
        ambiguous_patch = """diff --git a/a.txt b/a.txt
--- a/a.txt
+++ b/a.txt
@@ -1 +1 @@
-same
+changed
"""

        with self.assertRaisesRegex(GitDiffApplyError, "did not match any content"):
            apply_file_patch_to_text(first_file(missing_patch), "different\n")

        with self.assertRaisesRegex(GitDiffApplyError, "matched multiple locations at lines 1, 3"):
            apply_file_patch_to_text(first_file(ambiguous_patch), "same\nother\nsame\n")

    def test_apply_modify_patch_rejects_overlapping_hunks(self):
        patch = """diff --git a/a.txt b/a.txt
--- a/a.txt
+++ b/a.txt
@@ -1,2 +1,2 @@
 abc
-def
+DEF
@@ -2,2 +2,2 @@
 def
-ghi
+GHI
"""

        with self.assertRaisesRegex(GitDiffApplyError, "overlap"):
            apply_file_patch_to_text(first_file(patch), "abc\ndef\nghi\n")

    def test_apply_add_file_patch_builds_content(self):
        patch = """diff --git a/new.txt b/new.txt
new file mode 100644
--- /dev/null
+++ b/new.txt
@@ -0,0 +1,2 @@
+hello
+world
\\ No newline at end of file
"""

        result = apply_file_patch_to_text(first_file(patch), None)

        self.assertEqual(result.operation, "add")
        self.assertEqual(result.path, "new.txt")
        self.assertEqual(result.content, "hello\nworld")

    def test_apply_add_file_patch_rejects_existing_original_content(self):
        patch = """diff --git a/new.txt b/new.txt
--- /dev/null
+++ b/new.txt
@@ -0,0 +1 @@
+hello
"""

        with self.assertRaisesRegex(GitDiffApplyError, "already has original content"):
            apply_file_patch_to_text(first_file(patch), "existing\n")

    def test_apply_delete_file_patch_returns_delete_result_only_when_file_becomes_empty(self):
        patch = """diff --git a/old.txt b/old.txt
deleted file mode 100644
--- a/old.txt
+++ /dev/null
@@ -1,2 +0,0 @@
-old
-file
"""

        result = apply_file_patch_to_text(first_file(patch), "old\nfile\n")

        self.assertEqual(result.operation, "delete")
        self.assertEqual(result.path, "old.txt")
        self.assertIsNone(result.content)

    def test_apply_delete_file_patch_rejects_partial_delete(self):
        patch = """diff --git a/old.txt b/old.txt
--- a/old.txt
+++ /dev/null
@@ -1 +0,0 @@
-old
"""

        with self.assertRaisesRegex(GitDiffApplyError, "did not remove the entire file"):
            apply_file_patch_to_text(first_file(patch), "old\nfile\n")


if __name__ == "__main__":
    unittest.main()
