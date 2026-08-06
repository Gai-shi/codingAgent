from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from evals.context_compression_e2e.benchmark_case import create_case_workspace
from evals.context_compression_e2e.grader import grade_target


class ContextCompressionE2EGraderTest(unittest.TestCase):
    def test_grader_rejects_unimplemented_fixture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = create_case_workspace(Path(tmpdir), force=True).resolve()

            result = grade_target(target)

            self.assertFalse(result.passed)
            self.assertIn("DiffReviewTool implementation file exists", result.missing_required)

    def test_grader_rejects_legacy_contract_even_if_class_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = create_case_workspace(Path(tmpdir), force=True).resolve()
            self._write_legacy_implementation(target)

            result = grade_target(target)

            self.assertFalse(result.passed)
            self.assertIn("DiffReviewTool inherits SentinelToolBase", result.missing_required)
            self.assertIn("DiffReviewTool preserves config override retention marker", result.missing_required)
            self.assertTrue(any("legacy_registry" in hit for hit in result.forbidden_hits))

    @staticmethod
    def _write_legacy_implementation(target: Path) -> None:
        (target / "sentinel_lab" / "diff_review.py").write_text(
            textwrap.dedent(
                '''\
                """Wrong legacy implementation used when effective context is lost."""

                from __future__ import annotations

                from .legacy_registry import register


                class DiffReviewTool:
                    name = "diff_review"
                    CONTEXT_RETENTION_MARKER = "OBSOLETE-MARKER-0000"

                    def execute(self, **kwargs):
                        register("diff_review", self.execute)
                        return {"added_lines": 0, "deleted_lines": 0, "warnings": []}
                '''
            ),
            encoding="utf-8",
        )
        (target / "sentinel_lab" / "bootstrap.py").write_text(
            textwrap.dedent(
                '''\
                """Tool bootstrap module."""

                from __future__ import annotations

                from .core import CommandVault
                from .diff_review import DiffReviewTool
                from .legacy_registry import register


                def install_default_tools(vault: CommandVault) -> CommandVault:
                    register("diff_review", DiffReviewTool().execute)
                    return vault
                '''
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
