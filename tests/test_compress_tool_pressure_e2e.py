from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evals.compress_tool_pressure_e2e.benchmark_case import (
    CASE_CONFLICT_CONTRACT_DELAY,
    CASE_TRACE_DEBUG_DELAY,
    build_prompt_turns,
    create_case_workspace,
    noisy_evidence_text,
    prompt_stats,
)
from evals.compress_tool_pressure_e2e.grader import grade_target
from evals.compress_tool_pressure_e2e.run_ai_job_ab import (
    GUIDED_COMPRESS_TOOL_HINT,
    TOOL_POLICY_GUIDED,
    TOOL_POLICY_NEUTRAL,
    _compare,
    _effective_prompt_texts,
    _summarize_policy_results,
    _summarize_suite,
    collect_run_diagnostics,
)


class CompressToolPressureE2ETest(unittest.TestCase):
    def test_conflict_fixture_keeps_compatible_default_entrypoint(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = create_case_workspace(Path(tmp_dir), force=True, noise_blocks=3)

            index = (target / "evidence" / "00_index.txt").read_text(encoding="utf-8")
            default_packet = (target / "evidence" / "01_default_release_packet.txt").read_text(
                encoding="utf-8"
            )
            errata = (target / "evidence" / "97_contract_errata_index.txt").read_text(encoding="utf-8")
            decisions = (target / "evidence" / "contract_decisions.txt").read_text(encoding="utf-8")
            overrides = (target / "evidence" / "release_overrides.txt").read_text(encoding="utf-8")
            delta = (target / "evidence" / "final_contract_delta.txt").read_text(encoding="utf-8")
            candidate_delta = (target / "evidence" / "release_candidate_delta.txt").read_text(
                encoding="utf-8"
            )
            report = (target / "auditor" / "report.py").read_text(encoding="utf-8")
            manifest = (target / ".ai_job_eval_case.json").read_text(encoding="utf-8")

        self.assertIn("01_default_release_packet.txt", index)
        self.assertIn("ERRATA_ROUTE = evidence/97_contract_errata_index.txt", default_packet)
        self.assertIn("legacy_contract_archive.txt", errata)
        self.assertIn("KEEP-COMPRESS-TOOL-9173", decisions)
        self.assertIn("OBSOLETE_MARKER_0001", decisions)
        self.assertIn("compress-tool-preserved", overrides)
        self.assertIn("Final Contract Delta", delta)
        self.assertIn("KEEP-CANDIDATE-DEFAULT", candidate_delta)
        self.assertIn("return {}", report)
        self.assertIn(CASE_CONFLICT_CONTRACT_DELAY, manifest)

    def test_trace_debug_fixture_contains_delayed_trace_rule(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = create_case_workspace(
                Path(tmp_dir),
                force=True,
                noise_blocks=2,
                case_id=CASE_TRACE_DEBUG_DELAY,
                pressure="smoke",
            )

            index = (target / "evidence" / "00_index.txt").read_text(encoding="utf-8")
            triage = (target / "evidence" / "01_incident_triage_packet.log").read_text(
                encoding="utf-8"
            )
            errata = (target / "evidence" / "97_trace_errata_index.txt").read_text(encoding="utf-8")
            production_trace = (target / "evidence" / "production_trace.log").read_text(encoding="utf-8")
            notes = (target / "evidence" / "state_machine_notes.txt").read_text(encoding="utf-8")
            manifest = (target / "evidence" / "active_trace_manifest.txt").read_text(encoding="utf-8")
            candidate_manifest = (target / "evidence" / "replay_analysis_manifest.txt").read_text(
                encoding="utf-8"
            )
            decision = (target / "reconciler" / "decision.py").read_text(encoding="utf-8")

        self.assertIn("01_incident_triage_packet.log", index)
        self.assertIn("ERRATA_ROUTE = evidence/97_trace_errata_index.txt", triage)
        self.assertIn("incident_trace_archive.log", errata)
        self.assertIn("TRACE-KEEP-4821", production_trace)
        self.assertIn("TRACE-OBSOLETE-0001", production_trace)
        self.assertIn("open-manual-review", notes)
        self.assertIn("quarantine-ledger-batch", notes)
        self.assertIn("defer-audit-sync", notes)
        self.assertIn("Active Trace Manifest", manifest)
        self.assertIn("TRACE-CANDIDATE", candidate_manifest)
        self.assertIn('"action": "monitor"', decision)

    def test_prompts_do_not_name_compress_tool(self):
        for case_id in (CASE_CONFLICT_CONTRACT_DELAY, CASE_TRACE_DEBUG_DELAY):
            turns = build_prompt_turns(case_id=case_id, pressure="hard")
            joined = "\n".join(turn.text for turn in turns)

            self.assertNotIn("compress_tool", joined)
            self.assertNotIn("压缩", joined)
            self.assertIn("evidence/00_index.txt", joined)

    def test_guided_policy_injects_tool_hint_only_for_enabled_variant(self):
        turns = build_prompt_turns(case_id=CASE_CONFLICT_CONTRACT_DELAY, pressure="hard")

        neutral_enabled = _effective_prompt_texts(
            turns,
            tool_policy=TOOL_POLICY_NEUTRAL,
            disable_compress_tool=False,
        )
        guided_disabled = _effective_prompt_texts(
            turns,
            tool_policy=TOOL_POLICY_GUIDED,
            disable_compress_tool=True,
        )
        guided_enabled = _effective_prompt_texts(
            turns,
            tool_policy=TOOL_POLICY_GUIDED,
            disable_compress_tool=False,
        )

        self.assertNotIn("compress_tool", "\n".join(neutral_enabled))
        self.assertNotIn("compress_tool", "\n".join(guided_disabled))
        self.assertIn(GUIDED_COMPRESS_TOOL_HINT, guided_enabled[0])
        self.assertIn("compress_tool", "\n".join(guided_enabled))
        self.assertEqual(len(guided_enabled), len(turns))

    def test_conflict_grader_accepts_exact_hidden_contract_and_rejects_unimplemented_fixture(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = create_case_workspace(Path(tmp_dir), force=True, noise_blocks=1)

            failed = grade_target(target, run_tests=False)
            self.assertFalse(failed.passed)
            self.assertIn("preserves retention marker: KEEP-COMPRESS-TOOL-9173", failed.missing_required)

            (target / "auditor" / "report.py").write_text(
                '''"""Final report."""

from __future__ import annotations


def build_report() -> dict[str, object]:
    return {
        "title": "Q4-COMPRESS-AUDIT",
        "marker": "KEEP-COMPRESS-TOOL-9173",
        "policy_code": "POLICY-COMPRESS-42",
        "owner": "context-quality",
        "status": "ready-for-review",
        "summary": "compress-tool-preserved KEEP-COMPRESS-TOOL-9173",
        "risk_level": "elevated",
        "review_window": "2026-W33",
        "escalation_channel": "audit-war-room",
        "regions": {"primary": "iad-7", "secondary": "pdx-2"},
        "control_plane": "cp-orion-7",
        "contract_version": "ccdl-2026.08.17",
        "dataset_fingerprint": "fp-9ab4c2-771",
        "routing_keys": ["audit.q4.retention", "policy.compress.42", "marker.keep.9173"],
        "evidence_checksums": {
            "core": "sha256:ccdl-core-9173",
            "override": "sha256:hotfix-42-w33",
            "delta": "sha256:delta-final-771",
        },
        "audit_tags": ["contract-lock", "manual-review", "q4-retention"],
        "validation_gates": [
            "schema-freeze",
            "owner-ack",
            "marker-retention",
            "summary-prefix",
        ],
        "owner_chain": ["context-quality", "audit-platform", "release-ops"],
        "approval_chain": ["rfc-4172", "sec-1180", "ops-9301"],
        "runbook_steps": [
            "freeze-schema",
            "notify-context-quality",
            "verify-marker-retention",
            "publish-war-room-note",
        ],
        "watchlist": [
            "legacy-policy-loader",
            "summary-shadow-renderer",
            "owner-ack-cron",
        ],
        "notification_channels": ["#audit-war-room", "#context-quality"],
        "deadline_matrix": {
            "owner_ack": "2026-08-14T09:00Z",
            "schema_freeze": "2026-08-15T12:00Z",
            "release_gate": "2026-08-16T18:00Z",
        },
        "blocking_conditions": [
            "missing-owner-ack",
            "marker-mismatch",
            "policy-drift",
        ],
        "release_flags": {
            "requires_manual_review": "yes",
            "allow_legacy_policy": "no",
        },
    }
''',
                encoding="utf-8",
            )

            passed = grade_target(target)

        self.assertTrue(passed.passed, passed)

    def test_trace_grader_accepts_exact_hidden_rule(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = create_case_workspace(
                Path(tmp_dir),
                force=True,
                noise_blocks=1,
                case_id=CASE_TRACE_DEBUG_DELAY,
            )

            failed = grade_target(target, run_tests=False)
            self.assertFalse(failed.passed)
            self.assertIn("preserves action: open-manual-review", failed.missing_required)

            (target / "reconciler" / "decision.py").write_text(
                '''"""Build reconciliation plans for production trace events."""

from __future__ import annotations


def build_reconciliation_plan(event: dict[str, str]) -> dict[str, str]:
    try:
        attempts = int(event.get("attempts", "0"))
    except ValueError:
        attempts = 0

    if (
        event.get("tenant") == "aurora-ledger"
        and event.get("pipeline") == "delta-sync"
        and event.get("error_code") == "E-RETRY-9173"
        and attempts >= 3
    ):
        return {
            "action": "open-manual-review",
            "owner": "ledger-quality",
            "severity": "sev2",
            "retry_after_minutes": "0",
            "status": "blocked-on-ledger-review",
            "marker": event.get("trace_id", "TRACE-KEEP-4821"),
            "queue": "ledger-manual-review",
            "runbook": "rb-ledger-9173",
            "escalation_channel": "#ledger-quality",
            "sla": "PT0M",
            "evidence_hash": "sha256:trace-keep-4821",
            "decision_flags": ["manual-review", "retain-marker", "block-ledger"],
        }
    if (
        event.get("tenant") == "aurora-ledger"
        and event.get("pipeline") == "delta-sync"
        and event.get("error_code") == "E-CHECKSUM-7712"
        and attempts >= 1
    ):
        return {
            "action": "quarantine-ledger-batch",
            "owner": "ledger-integrity",
            "severity": "sev1",
            "retry_after_minutes": "0",
            "status": "blocked-on-integrity-check",
            "marker": event.get("trace_id", "TRACE-QUARANTINE-7712"),
            "queue": "ledger-quarantine",
            "runbook": "rb-ledger-7712",
            "escalation_channel": "#ledger-integrity",
            "sla": "PT0M",
            "evidence_hash": "sha256:trace-quarantine-7712",
            "decision_flags": ["quarantine", "checksum", "block-ledger"],
        }
    if (
        event.get("tenant") == "aurora-ledger"
        and event.get("pipeline") == "audit-sync"
        and event.get("error_code") == "E-AUDIT-LAG-3345"
        and attempts >= 2
    ):
        return {
            "action": "defer-audit-sync",
            "owner": "audit-quality",
            "severity": "sev3",
            "retry_after_minutes": "30",
            "status": "deferred-for-audit-window",
            "marker": event.get("trace_id", "TRACE-AUDIT-3345"),
            "queue": "audit-defer",
            "runbook": "rb-audit-3345",
            "escalation_channel": "#audit-quality",
            "sla": "PT30M",
            "evidence_hash": "sha256:trace-audit-3345",
            "decision_flags": ["audit-defer", "windowed", "retain-marker"],
        }
    return {
        "action": "monitor",
        "owner": "triage",
        "severity": "baseline",
        "retry_after_minutes": "15",
        "status": "watching",
        "marker": event.get("trace_id", ""),
        "queue": "triage-watch",
        "runbook": "rb-triage-default",
        "escalation_channel": "#triage",
        "sla": "PT15M",
        "evidence_hash": "sha256:default-monitor",
        "decision_flags": ["default", "non-active"],
    }
''',
                encoding="utf-8",
            )

            passed = grade_target(target)

        self.assertTrue(passed.passed, passed)

    def test_prompt_stats_reports_case_pressure_and_compatible_noise_size(self):
        turns = []

        stats = prompt_stats(turns, noise_blocks=2)

        self.assertEqual(stats["case_id"], CASE_CONFLICT_CONTRACT_DELAY)
        self.assertEqual(stats["pressure"], "hard")
        self.assertEqual(stats["noise_blocks"], 2)
        self.assertGreater(stats["evidence_chars"], len(noisy_evidence_text(noise_blocks=2)))

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
{"id": "call-read", "name": "read_file", "arguments": {"path": "evidence/contract_decisions.txt"}}
```

## 10:00:02 ToolResult read_file

```text
abcdefghijklmnopqrstuvwxyz
```

## 10:00:03 ToolCall compress_tool

```json
{"id": "call-compress", "name": "compress_tool", "arguments": {"replacements": [{"tool_name": "read_file", "tool_arguments": {"path": "evidence/contract_decisions.txt"}, "replace_content": "abc"}]}}
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

    def test_diagnostics_estimates_saved_chars_for_namespaced_replacement_tool_name(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            session_path = Path(tmp_dir) / "session.md"
            session_path.write_text(
                """# Session

## 10:00:01 ToolCall read_file

```json
{"id": "call-read", "name": "read_file", "arguments": {"path": "known.txt"}}
```

## 10:00:02 ToolResult read_file

```text
abcdefghijklmnopqrstuvwxyz
```

## 10:00:03 ToolCall compress_tool

```json
{"id": "call-compress", "name": "compress_tool", "arguments": {"replacements": [{"tool_name": "functions.read_file", "tool_arguments": {"path": "known.txt"}, "replace_content": "abc"}]}}
```

## 10:00:04 ToolResult compress_tool

```text
Success
```
""",
                encoding="utf-8",
            )
            stdout = f"session_record: {session_path}\n"

            diagnostics = collect_run_diagnostics(stdout, "")

        self.assertEqual(diagnostics["compress_tool_success_count"], 1)
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
{"id": "call-read", "name": "read_file", "arguments": {"path": "evidence/contract_decisions.txt"}}
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

    def test_summarize_suite_reports_correctness_and_efficiency_counters(self):
        cases = {
            "case_a": {
                "hard": {
                    "enabled": {"grade": {"passed": True, "score": 100}, "elapsed_seconds": 10},
                    "disabled": {"grade": {"passed": False, "score": 40}, "elapsed_seconds": 12},
                    "comparison": {
                        "compress_tool_helped": True,
                        "both_passed": False,
                        "both_failed": False,
                        "compress_tool_regressed": False,
                        "enabled_compress_tool_effective": True,
                        "enabled_compress_tool_call_count": 2,
                        "enabled_estimated_tool_context_chars_saved": 1000,
                    },
                }
            },
            "case_b": {
                "hard": {
                    "enabled": {"grade": {"passed": True, "score": 90}, "elapsed_seconds": 9},
                    "disabled": {"grade": {"passed": True, "score": 80}, "elapsed_seconds": 8},
                    "comparison": {
                        "compress_tool_helped": False,
                        "both_passed": True,
                        "both_failed": False,
                        "compress_tool_regressed": False,
                        "enabled_compress_tool_effective": False,
                        "enabled_compress_tool_call_count": 0,
                        "enabled_estimated_tool_context_chars_saved": 0,
                    },
                }
            },
        }

        summary = _summarize_suite(cases)

        self.assertEqual(summary["cell_count"], 2)
        self.assertEqual(summary["enabled_pass_count"], 2)
        self.assertEqual(summary["disabled_pass_count"], 1)
        self.assertEqual(summary["correctness_delta"], 1)
        self.assertEqual(summary["compress_tool_helped_count"], 1)
        self.assertEqual(summary["enabled_compress_tool_effective_count"], 1)
        self.assertEqual(summary["enabled_estimated_tool_context_chars_saved"], 1000)

    def test_summarize_policy_results_keeps_neutral_and_guided_deltas_separate(self):
        policy_results = {
            TOOL_POLICY_NEUTRAL: {
                "summary": {
                    "cell_count": 2,
                    "enabled_pass_count": 1,
                    "disabled_pass_count": 1,
                    "correctness_delta": 0,
                    "enabled_score_total": 140,
                    "disabled_score_total": 140,
                    "score_delta_total": 0,
                    "compress_tool_helped_count": 0,
                    "both_passed_count": 1,
                    "both_failed_count": 1,
                    "compress_tool_regressed_count": 0,
                    "enabled_compress_tool_effective_count": 0,
                    "enabled_compress_tool_call_count": 0,
                    "enabled_estimated_tool_context_chars_saved": 0,
                    "enabled_elapsed_seconds": 10,
                    "disabled_elapsed_seconds": 9,
                }
            },
            TOOL_POLICY_GUIDED: {
                "summary": {
                    "cell_count": 2,
                    "enabled_pass_count": 2,
                    "disabled_pass_count": 1,
                    "correctness_delta": 1,
                    "enabled_score_total": 190,
                    "disabled_score_total": 140,
                    "score_delta_total": 50,
                    "compress_tool_helped_count": 1,
                    "both_passed_count": 1,
                    "both_failed_count": 0,
                    "compress_tool_regressed_count": 0,
                    "enabled_compress_tool_effective_count": 1,
                    "enabled_compress_tool_call_count": 2,
                    "enabled_estimated_tool_context_chars_saved": 1000,
                    "enabled_elapsed_seconds": 12,
                    "disabled_elapsed_seconds": 9,
                }
            },
        }

        summary = _summarize_policy_results(policy_results)

        self.assertEqual(summary["policy_count"], 2)
        self.assertEqual(summary["cell_count"], 4)
        self.assertEqual(summary["correctness_delta"], 1)
        self.assertEqual(summary["compress_tool_helped_count"], 1)
        self.assertEqual(summary["pure_availability_delta"], 0)
        self.assertEqual(summary["guided_tool_delta"], 1)
        self.assertEqual(summary["enabled_compress_tool_call_count"], 2)


if __name__ == "__main__":
    unittest.main()
