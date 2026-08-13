from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evals.compress_tool_pressure_e2e.benchmark_case import (
    create_case_workspace,
    noisy_evidence_text,
    prompt_stats,
)
from evals.compress_tool_pressure_e2e.grader import grade_target
from evals.compress_tool_pressure_e2e.run_ai_job_ab import (
    _compare,
    collect_run_diagnostics,
)


class CompressToolPressureE2ETest(unittest.TestCase):
    def test_fixture_contains_large_noisy_evidence_and_unimplemented_report(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = create_case_workspace(Path(tmp_dir), force=True, noise_blocks=3)

            evidence = (target / "evidence" / "noisy_audit_log.txt").read_text(encoding="utf-8")
            report = (target / "auditor" / "report.py").read_text(encoding="utf-8")

        self.assertIn("KEEP-COMPRESS-TOOL-9173", evidence)
        self.assertIn("OBSOLETE_MARKER_0001", evidence)
        self.assertIn("return {}", report)

    def test_grader_accepts_canonical_report_and_rejects_unimplemented_fixture(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = create_case_workspace(Path(tmp_dir), force=True, noise_blocks=1)

            failed = grade_target(target, run_tests=False)
            self.assertFalse(failed.passed)
            self.assertIn("preserves retention marker: KEEP-COMPRESS-TOOL-9173", failed.missing_required)

            (target / "auditor" / "report.py").write_text(
                '''"""Final report."""

from __future__ import annotations


def build_report() -> dict[str, str]:
    return {
        "title": "Q4-COMPRESS-AUDIT",
        "marker": "KEEP-COMPRESS-TOOL-9173",
        "policy_code": "POLICY-COMPRESS-42",
        "owner": "context-quality",
        "status": "ready-for-review",
        "summary": "compress-tool-preserved final facts",
    }
''',
                encoding="utf-8",
            )

            passed = grade_target(target)

        self.assertTrue(passed.passed, passed)

    def test_prompt_stats_reports_evidence_pressure_size(self):
        turns = []

        stats = prompt_stats(turns, noise_blocks=2)

        self.assertEqual(stats["noise_blocks"], 2)
        self.assertEqual(stats["evidence_chars"], len(noisy_evidence_text(noise_blocks=2)))

    def test_diagnostics_count_compress_tool_usage_and_estimated_saved_chars(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            session_path = Path(tmp_dir) / "session.md"
            session_path.write_text(
                """# Session

## 10:00:00 UserMessage

```text
请完成当前 workspace 里的 auditor 报告实现
```

## 10:00:01 ToolCall read_file

```json
{"id": "call-read", "name": "read_file", "arguments": {"path": "evidence/noisy_audit_log.txt"}}
```

## 10:00:02 ToolResult read_file

```text
abcdefghijklmnopqrstuvwxyz
```

## 10:00:03 ToolCall compress_tool

```json
{"id": "call-compress", "name": "compress_tool", "arguments": {"replacements": [{"tool_name": "read_file", "tool_arguments": {"path": "evidence/noisy_audit_log.txt"}, "replace_content": "abc"}]}}
```

## 10:00:04 ToolResult compress_tool

```text
Success
```
""",
                encoding="utf-8",
            )
            stdout = f"session_record: {session_path}\nlog_file: /tmp/log.log\n"

            diagnostics = collect_run_diagnostics(stdout, "")

        self.assertEqual(diagnostics["read_file_tool_call_count"], 1)
        self.assertEqual(diagnostics["compress_tool_call_count"], 1)
        self.assertEqual(diagnostics["compress_tool_success_count"], 1)
        self.assertEqual(diagnostics["read_file_tool_result_chars"], 26)
        self.assertEqual(diagnostics["largest_read_file_tool_result_chars"], 26)
        self.assertEqual(diagnostics["compress_replacement_chars"], 3)
        self.assertEqual(diagnostics["successful_compress_replacement_chars"], 3)
        self.assertEqual(diagnostics["estimated_tool_context_chars_saved"], 23)

    def test_diagnostics_counts_large_tool_results_without_cross_section_regex_drift(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            session_path = Path(tmp_dir) / "session.md"
            large_output = "x" * 5000
            session_path.write_text(
                f"""# Session

## 10:00:00 SystemMessage

```text
system prompt
```

## 10:00:01 ToolCall read_file

```json
{{"id": "call-large", "name": "read_file", "arguments": {{"path": "large.txt"}}}}
```

## 10:00:02 ToolResult read_file

```text
{large_output}
```

## 10:00:03 ToolCall read_file

```json
{{"id": "call-small", "name": "read_file", "arguments": {{"path": "small.txt"}}}}
```

## 10:00:04 ToolResult read_file

```text
small
```
""",
                encoding="utf-8",
            )
            stdout = f"session_record: {session_path}\n"

            diagnostics = collect_run_diagnostics(stdout, "")

        self.assertEqual(diagnostics["read_file_tool_result_chars"], 5005)
        self.assertEqual(diagnostics["largest_read_file_tool_result_chars"], 5000)

    def test_diagnostics_reports_failed_compress_tool_attempt_without_saved_chars(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            session_path = Path(tmp_dir) / "session.md"
            session_path.write_text(
                """# Session

## 10:00:01 ToolCall read_file

```json
{"id": "call-read", "name": "read_file", "arguments": {"path": "evidence/noisy_audit_log.txt"}}
```

## 10:00:02 ToolResult read_file

```text
abcdefghijklmnopqrstuvwxyz
```

## 10:00:03 ToolCall compress_tool

```json
{"id": "call-compress", "name": "compress_tool", "arguments": {"replacements": [{"tool_name": "read_file", "tool_arguments": {"path": "missing.txt"}, "replace_content": "abc"}]}}
```

## 10:00:04 ToolResult compress_tool

```text
Error: no previous tool result matches read_file with arguments {"path":"missing.txt"}
```
""",
                encoding="utf-8",
            )
            stdout = f"session_record: {session_path}\n"

            diagnostics = collect_run_diagnostics(stdout, "")

        self.assertEqual(diagnostics["compress_tool_success_count"], 0)
        self.assertEqual(diagnostics["compress_tool_error_count"], 1)
        self.assertEqual(
            diagnostics["compress_tool_errors"],
            ['Error: no previous tool result matches read_file with arguments {"path":"missing.txt"}'],
        )
        self.assertEqual(diagnostics["compress_replacement_chars"], 3)
        self.assertEqual(diagnostics["successful_compress_replacement_chars"], 0)
        self.assertEqual(diagnostics["estimated_tool_context_chars_saved"], 0)

    def test_compare_marks_helped_only_when_enabled_passes_disabled_fails_and_compress_succeeds(self):
        enabled = {
            "grade": {"passed": True, "score": 100},
            "diagnostics": {
                "compress_tool_call_count": 1,
                "compress_tool_success_count": 1,
                "compress_tool_error_count": 0,
                "estimated_tool_context_chars_saved": 1000,
            },
        }
        disabled = {
            "grade": {"passed": False, "score": 40},
            "diagnostics": {"compress_tool_call_count": 0},
        }

        comparison = _compare(enabled, disabled)

        self.assertTrue(comparison["compress_tool_helped"])
        self.assertFalse(comparison["both_passed"])
        self.assertEqual(comparison["verdict"], "compress_tool_helped")
        self.assertEqual(comparison["enabled_score"], 100)
        self.assertEqual(comparison["disabled_score"], 40)

    def test_compare_marks_both_passed_with_failed_compress_as_inconclusive(self):
        enabled = {
            "grade": {"passed": True, "score": 100},
            "diagnostics": {
                "compress_tool_call_count": 1,
                "compress_tool_success_count": 0,
                "compress_tool_error_count": 1,
                "estimated_tool_context_chars_saved": 0,
            },
        }
        disabled = {
            "grade": {"passed": True, "score": 100},
            "diagnostics": {"compress_tool_call_count": 0},
        }

        comparison = _compare(enabled, disabled)

        self.assertFalse(comparison["compress_tool_helped"])
        self.assertTrue(comparison["both_passed"])
        self.assertEqual(comparison["verdict"], "inconclusive_enabled_compress_tool_failed")


if __name__ == "__main__":
    unittest.main()
