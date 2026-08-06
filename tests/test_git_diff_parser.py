from __future__ import annotations

import unittest

from ai_job.tools.git_diff_parser import GitDiffParseError, HunkLine, parse_git_diff


class GitDiffParserTest(unittest.TestCase):
    def test_parse_modify_file_patch(self):
        patch = """diff --git a/src/calc.py b/src/calc.py
index 1111111..2222222 100644
--- a/src/calc.py
+++ b/src/calc.py
@@ -1,2 +1,2 @@ def add
 def add(a, b):
-    return a - b
+    return a + b
"""

        parsed = parse_git_diff(patch)

        self.assertEqual(len(parsed.files), 1)
        file_patch = parsed.files[0]
        self.assertEqual(file_patch.operation, "modify")
        self.assertEqual(file_patch.old_path, "src/calc.py")
        self.assertEqual(file_patch.new_path, "src/calc.py")
        self.assertEqual(len(file_patch.hunks), 1)
        hunk = file_patch.hunks[0]
        self.assertEqual((hunk.old_start, hunk.old_count, hunk.new_start, hunk.new_count), (1, 2, 1, 2))
        self.assertEqual(hunk.section_header, "def add")
        self.assertEqual(
            hunk.lines,
            [
                HunkLine(kind="context", text="def add(a, b):"),
                HunkLine(kind="remove", text="    return a - b"),
                HunkLine(kind="add", text="    return a + b"),
            ],
        )

    def test_parse_add_file_patch_with_no_newline_marker(self):
        patch = """diff --git a/src/new.py b/src/new.py
new file mode 100644
index 0000000..1111111
--- /dev/null
+++ b/src/new.py
@@ -0,0 +1,2 @@
+hello
+world
\\ No newline at end of file
"""

        parsed = parse_git_diff(patch)

        file_patch = parsed.files[0]
        self.assertEqual(file_patch.operation, "add")
        self.assertIsNone(file_patch.old_path)
        self.assertEqual(file_patch.new_path, "src/new.py")
        self.assertEqual(
            file_patch.hunks[0].lines,
            [
                HunkLine(kind="add", text="hello"),
                HunkLine(kind="add", text="world", new_no_newline=True),
            ],
        )

    def test_parse_delete_file_patch(self):
        patch = """diff --git a/src/old.py b/src/old.py
deleted file mode 100644
index 1111111..0000000
--- a/src/old.py
+++ /dev/null
@@ -1,2 +0,0 @@
-old
-file
"""

        parsed = parse_git_diff(patch)

        file_patch = parsed.files[0]
        self.assertEqual(file_patch.operation, "delete")
        self.assertEqual(file_patch.old_path, "src/old.py")
        self.assertIsNone(file_patch.new_path)
        self.assertEqual(
            file_patch.hunks[0].lines,
            [
                HunkLine(kind="remove", text="old"),
                HunkLine(kind="remove", text="file"),
            ],
        )

    def test_parse_multiple_file_patches(self):
        patch = """diff --git a/a.txt b/a.txt
--- a/a.txt
+++ b/a.txt
@@ -1 +1 @@
-old
+new
diff --git a/b.txt b/b.txt
--- /dev/null
+++ b/b.txt
@@ -0,0 +1 @@
+created
"""

        parsed = parse_git_diff(patch)

        self.assertEqual([file.operation for file in parsed.files], ["modify", "add"])
        self.assertEqual([file.new_path for file in parsed.files], ["a.txt", "b.txt"])

    def test_rejects_unsupported_rename_and_quoted_paths(self):
        rename_patch = """diff --git a/old.py b/new.py
similarity index 100%
rename from old.py
rename to new.py
"""
        quoted_patch = 'diff --git "a/path with space.py" "b/path with space.py"\n'

        with self.assertRaisesRegex(GitDiffParseError, "unsupported git diff header"):
            parse_git_diff(rename_patch)
        with self.assertRaisesRegex(GitDiffParseError, "unsupported diff header"):
            parse_git_diff(quoted_patch)

    def test_rejects_hunk_count_mismatch(self):
        patch = """diff --git a/a.txt b/a.txt
--- a/a.txt
+++ b/a.txt
@@ -1,2 +1,1 @@
-old
+new
"""

        with self.assertRaisesRegex(GitDiffParseError, "declares 2 old lines but contains 1"):
            parse_git_diff(patch)


if __name__ == "__main__":
    unittest.main()
