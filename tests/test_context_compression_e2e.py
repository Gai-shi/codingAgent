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

    def test_grader_accepts_canonical_multi_file_implementation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = create_case_workspace(Path(tmpdir), force=True).resolve()
            self._write_canonical_implementation(target)

            result = grade_target(target)

            self.assertTrue(result.passed, result)
            self.assertIn("DiffReviewTool uses approved audit topology path", result.required_hits)
            self.assertIn("warning policy preserves code: W-MARCH-TODO-214", result.required_hits)

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

    @staticmethod
    def _write_canonical_implementation(target: Path) -> None:
        audit_dir = target / "sentinel_lab" / "audit"
        (audit_dir / "warning_policy.py").write_text(
            textwrap.dedent(
                '''\
                from __future__ import annotations

                POLICY_VERSION = "MARCH-AUDIT-V7"
                WARNING_FILE_MISMATCH = "W-MARCH-FILE-337"
                WARNING_TODO = "W-MARCH-TODO-214"
                ERROR_STRICT = "E-MARCH-STRICT-901"
                ERROR_EMPTY_PATCH = "E-MARCH-EMPTY-044"


                def build_warning(code: str, message: str) -> dict[str, str]:
                    return {"code": code, "message": message, "severity": "warning"}
                '''
            ),
            encoding="utf-8",
        )
        (audit_dir / "unified_diff_parser.py").write_text(
            textwrap.dedent(
                '''\
                from __future__ import annotations


                def parse_unified_diff(patch_text: str) -> dict[str, object]:
                    added_lines = 0
                    deleted_lines = 0
                    changed_files: set[str] = set()
                    todo_found = False
                    for line in patch_text.splitlines():
                        if line.startswith("+++ "):
                            path = normalize_diff_path(line[4:].strip().split(maxsplit=1)[0])
                            if path and path != "/dev/null":
                                changed_files.add(path)
                            continue
                        if line.startswith("--- "):
                            path = normalize_diff_path(line[4:].strip().split(maxsplit=1)[0])
                            if path and path != "/dev/null":
                                changed_files.add(path)
                            continue
                        if line.startswith("+"):
                            added_lines += 1
                            if "TODO" in line[1:]:
                                todo_found = True
                        elif line.startswith("-"):
                            deleted_lines += 1
                    return {
                        "added_lines": added_lines,
                        "deleted_lines": deleted_lines,
                        "changed_files": sorted(changed_files),
                        "todo_found": todo_found,
                    }


                def normalize_diff_path(path: str) -> str:
                    if path.startswith("a/") or path.startswith("b/"):
                        return path[2:]
                    return path
                '''
            ),
            encoding="utf-8",
        )
        (audit_dir / "diff_review_tool.py").write_text(
            textwrap.dedent(
                '''\
                from __future__ import annotations

                from .unified_diff_parser import normalize_diff_path, parse_unified_diff
                from .warning_policy import (
                    ERROR_EMPTY_PATCH,
                    ERROR_STRICT,
                    POLICY_VERSION,
                    WARNING_FILE_MISMATCH,
                    WARNING_TODO,
                    build_warning,
                )
                from ..core import GuardedToolOutcome, MarchConfig, SentinelToolBase


                class DiffReviewTool(SentinelToolBase):
                    name = "diff_review"
                    CONTEXT_RETENTION_MARKER = "MARCH-CONTEXT-7429"
                    TOPOLOGY_RETENTION_MARKER = "MARCH-TOPOLOGY-3190"
                    CONFIG_RETENTION_MARKER = "MARCH-CONFIG-5812"
                    POLICY_RETENTION_MARKER = "MARCH-POLICY-2664"

                    def __init__(self, config: MarchConfig | None = None) -> None:
                        super().__init__(config=config)

                    def execute(self, file_path: str, patch_text: str, strict: bool = False) -> GuardedToolOutcome:
                        if not patch_text or not patch_text.strip():
                            return GuardedToolOutcome.fail(f"{ERROR_EMPTY_PATCH}: patch_text is empty")
                        parsed = parse_unified_diff(patch_text)
                        warnings: list[dict[str, str]] = []
                        changed_files = parsed["changed_files"]
                        if normalize_diff_path(file_path) not in changed_files:
                            warnings.append(build_warning(WARNING_FILE_MISMATCH, f"{file_path} not found in diff headers"))
                        if parsed["todo_found"]:
                            warnings.append(build_warning(WARNING_TODO, "TODO found in added lines"))
                        payload = {
                            "added_lines": parsed["added_lines"],
                            "deleted_lines": parsed["deleted_lines"],
                            "changed_files": changed_files,
                            "warnings": warnings,
                            "policy_version": POLICY_VERSION,
                        }
                        if strict and warnings:
                            return GuardedToolOutcome.fail(f"{ERROR_STRICT}: strict warnings", payload=payload)
                        return GuardedToolOutcome.success(payload)
                '''
            ),
            encoding="utf-8",
        )
        (audit_dir / "__init__.py").write_text(
            "from .diff_review_tool import DiffReviewTool\n\n__all__ = [\"DiffReviewTool\"]\n",
            encoding="utf-8",
        )
        (target / "sentinel_lab" / "bootstrap.py").write_text(
            textwrap.dedent(
                '''\
                from __future__ import annotations

                from .audit import DiffReviewTool
                from .core import CommandVault, MarchConfig


                def install_default_tools(vault: CommandVault) -> CommandVault:
                    vault.install(DiffReviewTool(MarchConfig(audit_label="march-diff-review", policy_version="MARCH-AUDIT-V7")))
                    return vault
                '''
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
