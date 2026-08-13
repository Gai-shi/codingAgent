from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_job.tools.apply_patch_tool import apply_patch_text


class ApplyPatchToolTest(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self._tmp_dir.name).resolve()

    def tearDown(self):
        self._tmp_dir.cleanup()

    def _write_text(self, relative_path: str, text: str) -> Path:
        path = self.workspace_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_apply_patch_text_modifies_adds_and_deletes_files(self):
        self._write_text("src/calc.py", "def add(a, b):\n    return a - b\n")
        self._write_text("src/old.py", "old\nfile\n")
        patch = """diff --git a/src/calc.py b/src/calc.py
--- a/src/calc.py
+++ b/src/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
diff --git a/tests/test_calc.py b/tests/test_calc.py
new file mode 100644
--- /dev/null
+++ b/tests/test_calc.py
@@ -0,0 +1,2 @@
+def test_add():
+    assert True
diff --git a/src/old.py b/src/old.py
deleted file mode 100644
--- a/src/old.py
+++ /dev/null
@@ -1,2 +0,0 @@
-old
-file
"""

        result = apply_patch_text({"patch": patch}, self.workspace_root)

        self.assertIn("Successfully applied patch: 3 file(s) changed, 3 hunk(s).", result)
        self.assertEqual((self.workspace_root / "src/calc.py").read_text(encoding="utf-8"), "def add(a, b):\n    return a + b\n")
        self.assertEqual((self.workspace_root / "tests/test_calc.py").read_text(encoding="utf-8"), "def test_add():\n    assert True\n")
        self.assertFalse((self.workspace_root / "src/old.py").exists())

    def test_apply_patch_text_rejects_existing_new_file_without_overwriting(self):
        self._write_text("new.txt", "existing\n")
        patch = """diff --git a/new.txt b/new.txt
new file mode 100644
--- /dev/null
+++ b/new.txt
@@ -0,0 +1 @@
+created
"""

        with self.assertRaisesRegex(FileExistsError, "new file already exists"):
            apply_patch_text({"patch": patch}, self.workspace_root)

        self.assertEqual((self.workspace_root / "new.txt").read_text(encoding="utf-8"), "existing\n")

    def test_apply_patch_text_prechecks_all_files_before_writing_anything(self):
        self._write_text("ok.txt", "old\n")
        patch = """diff --git a/ok.txt b/ok.txt
--- a/ok.txt
+++ b/ok.txt
@@ -1 +1 @@
-old
+new
diff --git a/missing.txt b/missing.txt
--- a/missing.txt
+++ b/missing.txt
@@ -1 +1 @@
-missing
+changed
"""

        with self.assertRaisesRegex(FileNotFoundError, "missing.txt"):
            apply_patch_text({"patch": patch}, self.workspace_root)

        self.assertEqual((self.workspace_root / "ok.txt").read_text(encoding="utf-8"), "old\n")

    def test_apply_patch_text_rejects_protected_and_escaping_paths(self):
        protected_patch = """diff --git a/.env b/.env
--- a/.env
+++ b/.env
@@ -1 +1 @@
-old
+new
"""
        escape_patch = """diff --git a/../outside.txt b/../outside.txt
--- a/../outside.txt
+++ b/../outside.txt
@@ -1 +1 @@
-old
+new
"""

        with self.assertRaisesRegex(PermissionError, "protected path"):
            apply_patch_text({"patch": protected_patch}, self.workspace_root)
        with self.assertRaisesRegex(ValueError, "escapes workspace"):
            apply_patch_text({"patch": escape_patch}, self.workspace_root)

    def test_apply_patch_text_rejects_duplicate_file_diffs_before_writing(self):
        self._write_text("same.txt", "old\n")
        patch = """diff --git a/same.txt b/same.txt
--- a/same.txt
+++ b/same.txt
@@ -1 +1 @@
-old
+new
diff --git a/same.txt b/same.txt
--- a/same.txt
+++ b/same.txt
@@ -1 +1 @@
-new
+newer
"""

        with self.assertRaisesRegex(ValueError, "multiple file diffs for the same path"):
            apply_patch_text({"patch": patch}, self.workspace_root)

        self.assertEqual((self.workspace_root / "same.txt").read_text(encoding="utf-8"), "old\n")

    def test_apply_patch_text_rejects_invalid_arguments(self):
        with self.assertRaisesRegex(ValueError, '"patch" must be a string'):
            apply_patch_text({"patch": None}, self.workspace_root)


if __name__ == "__main__":
    unittest.main()
