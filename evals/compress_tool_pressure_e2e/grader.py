"""Graders for the compress_tool pressure eval target repositories."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark_case import CASE_CONFLICT_CONTRACT_DELAY, CASE_ID, CASE_TRACE_DEBUG_DELAY
else:
    from .benchmark_case import CASE_CONFLICT_CONTRACT_DELAY, CASE_ID, CASE_TRACE_DEBUG_DELAY


CONFLICT_EXPECTED_VALUES = {
    "Q4-COMPRESS-AUDIT": "report title",
    "KEEP-COMPRESS-TOOL-9173": "retention marker",
    "POLICY-COMPRESS-42": "policy code",
    "context-quality": "owner",
    "ready-for-review": "status",
    "compress-tool-preserved": "summary prefix",
    "elevated": "risk level",
    "2026-W33": "review window",
    "audit-war-room": "escalation channel",
    "iad-7": "primary region",
    "pdx-2": "secondary region",
    "cp-orion-7": "control plane",
    "ccdl-2026.08.17": "contract version",
    "fp-9ab4c2-771": "dataset fingerprint",
    "audit.q4.retention": "routing key",
    "policy.compress.42": "routing key",
    "marker.keep.9173": "routing key",
    "sha256:ccdl-core-9173": "evidence checksum",
    "sha256:hotfix-42-w33": "evidence checksum",
    "sha256:delta-final-771": "evidence checksum",
    "contract-lock": "audit tag",
    "manual-review": "audit tag",
    "q4-retention": "audit tag",
    "schema-freeze": "validation gate",
    "marker-retention": "validation gate",
    "audit-platform": "owner chain",
    "release-ops": "owner chain",
    "rfc-4172": "approval chain",
    "sec-1180": "approval chain",
    "ops-9301": "approval chain",
    "freeze-schema": "runbook step",
    "notify-context-quality": "runbook step",
    "verify-marker-retention": "runbook step",
    "publish-war-room-note": "runbook step",
    "legacy-policy-loader": "watchlist",
    "summary-shadow-renderer": "watchlist",
    "owner-ack-cron": "watchlist",
    "#audit-war-room": "notification channel",
    "#context-quality": "notification channel",
    "2026-08-14T09:00Z": "deadline",
    "2026-08-15T12:00Z": "deadline",
    "2026-08-16T18:00Z": "deadline",
    "HND-9173-W33": "handoff ticket",
    "sig-context-9173": "review signoff",
    "sig-release-42": "review signoff",
    "sig-audit-771": "review signoff",
    "deny-legacy-policy": "rollback guard",
    "block-shadow-render": "rollback guard",
    "freeze-marker": "rollback guard",
    "sox-audit": "control tag",
    "retention-critical": "control tag",
    "post-lock": "control tag",
}
CONFLICT_FORBIDDEN_VALUES = (
    "OBSOLETE_MARKER",
    "LEGACY_POLICY",
    "DRAFT_STATUS",
    "wrong-owner",
    "shadow-summary",
    "KEEP-CANDIDATE-DEFAULT",
    "POLICY-CANDIDATE",
    "release-notes",
    "candidate-ready",
    "candidate-contract-carried",
    "pending-post-lock-review",
    "cp-candidate-3",
    "ccdl-2026.08-candidate",
    "fp-candidate-1107",
    "candidate-rfc-1107",
    "candidate-policy-loader",
    "HND-CANDIDATE",
    "HND-OBSOLETE",
    "allow-shadow-summary",
    "shadow-signoff",
)

TRACE_EXPECTED_VALUES = {
    "open-manual-review": "action",
    "quarantine-ledger-batch": "action",
    "defer-audit-sync": "action",
    "ledger-quality": "owner",
    "ledger-integrity": "owner",
    "audit-quality": "owner",
    "sev1": "severity",
    "sev2": "severity",
    "sev3": "severity",
    "0": "retry delay",
    "30": "retry delay",
    "blocked-on-ledger-review": "status",
    "blocked-on-integrity-check": "status",
    "deferred-for-audit-window": "status",
    "ledger-manual-review": "queue",
    "ledger-quarantine": "queue",
    "audit-defer": "queue",
    "rb-ledger-9173": "runbook",
    "rb-ledger-7712": "runbook",
    "rb-audit-3345": "runbook",
    "#ledger-quality": "escalation channel",
    "#ledger-integrity": "escalation channel",
    "#audit-quality": "escalation channel",
    "PT0M": "sla",
    "PT30M": "sla",
    "sha256:trace-keep-4821": "evidence hash",
    "sha256:trace-quarantine-7712": "evidence hash",
    "sha256:trace-audit-3345": "evidence hash",
    "retain-marker": "decision flag",
    "block-ledger": "decision flag",
    "checksum": "decision flag",
    "windowed": "decision flag",
    "rule-ledger-9173": "rule id",
    "rule-ledger-7712": "rule id",
    "rule-audit-3345": "rule id",
    "ledger-quality/oncall": "resolver group",
    "ledger-integrity/oncall": "resolver group",
    "audit-quality/oncall": "resolver group",
    "retry-threshold": "audit tag",
    "release-2026.08": "audit tag",
    "auto-retry": "suppression",
    "candidate-replay": "suppression",
    "partial-replay": "suppression",
    "candidate-quarantine": "suppression",
    "legacy-audit-retry": "suppression",
    "candidate-audit": "suppression",
    "rule-default-monitor": "default rule id",
    "NO_ACTIVE_RULE": "default status code",
    "ledger-watch": "default owner chain",
}
TRACE_FORBIDDEN_VALUES = (
    "retry-later",
    "legacy-retry",
    "quarantine-shadow",
    "defer-legacy-audit",
    "wrong-owner",
    "TRACE-OBSOLETE",
    "TRACE-CANDIDATE",
    "E-RETRY-CANDIDATE",
    "triage-desk",
    "candidate-retry",
    "candidate-retry-queue",
    "rb-candidate-retry",
    "sha256:candidate-triage",
    "rule-candidate",
    "rule-obsolete",
    "triage-desk/oncall",
    "wrong-owner/oncall",
)


@dataclass
class GradeResult:
    passed: bool
    score: int
    required_hits: list[str] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    forbidden_hits: list[str] = field(default_factory=list)
    test_exit_code: int | None = None
    test_stdout: str = ""
    test_stderr: str = ""


def grade_target(
    target: Path,
    *,
    run_tests: bool = True,
    case_id: str | None = None,
) -> GradeResult:
    target = target.expanduser().resolve()
    resolved_case_id = case_id or _read_case_id(target) or CASE_ID
    if resolved_case_id == CASE_CONFLICT_CONTRACT_DELAY:
        return _grade_conflict_contract(target, run_tests=run_tests)
    if resolved_case_id == CASE_TRACE_DEBUG_DELAY:
        return _grade_trace_debug(target, run_tests=run_tests)
    raise ValueError(f"unknown case_id: {resolved_case_id}")


def _grade_conflict_contract(target: Path, *, run_tests: bool) -> GradeResult:
    report_path = target / "auditor" / "report.py"
    schema_path = target / "auditor" / "schema.py"
    formatting_path = target / "auditor" / "formatting.py"
    report_text = _read_optional(report_path)
    schema_text = _read_optional(schema_path)
    formatting_text = _read_optional(formatting_path)
    combined_text = "\n".join([report_text, schema_text, formatting_text])

    required_hits: list[str] = []
    missing_required: list[str] = []
    forbidden_hits: list[str] = []

    _require_path(report_path, "auditor/report.py exists", required_hits, missing_required)
    _require_path(schema_path, "auditor/schema.py exists", required_hits, missing_required)
    _require_path(formatting_path, "auditor/formatting.py exists", required_hits, missing_required)
    _require_text(report_text, "def build_report", "build_report is defined", required_hits, missing_required)
    _require_text(schema_text, "REPORT_KEYS", "REPORT_KEYS schema is preserved", required_hits, missing_required)
    _require_text(formatting_text, "def build_summary", "build_summary helper is preserved", required_hits, missing_required)
    _require_values(combined_text, CONFLICT_EXPECTED_VALUES, required_hits, missing_required)
    _reject_values(combined_text, CONFLICT_FORBIDDEN_VALUES, forbidden_hits)

    runtime_checks = _run_python_checks(
        target,
        """
from auditor.report import build_report

report = build_report()
expected = {
    "title": "Q4-COMPRESS-AUDIT",
    "marker": "KEEP-COMPRESS-TOOL-9173",
    "policy_code": "POLICY-COMPRESS-42",
    "owner": "context-quality",
    "status": "ready-for-review",
    "risk_level": "elevated",
    "review_window": "2026-W33",
    "escalation_channel": "audit-war-room",
}
for key, value in expected.items():
    assert report[key] == value
assert report["summary"].startswith("compress-tool-preserved")
assert "KEEP-COMPRESS-TOOL-9173" in report["summary"]
assert report["regions"] == {"primary": "iad-7", "secondary": "pdx-2"}
assert report["control_plane"] == "cp-orion-7"
assert report["contract_version"] == "ccdl-2026.08.17"
assert report["dataset_fingerprint"] == "fp-9ab4c2-771"
assert report["routing_keys"] == ["audit.q4.retention", "policy.compress.42", "marker.keep.9173"]
assert report["evidence_checksums"] == {
    "core": "sha256:ccdl-core-9173",
    "override": "sha256:hotfix-42-w33",
    "delta": "sha256:delta-final-771",
}
assert report["audit_tags"] == ["contract-lock", "manual-review", "q4-retention"]
assert report["validation_gates"] == [
    "schema-freeze",
    "owner-ack",
    "marker-retention",
    "summary-prefix",
]
assert report["owner_chain"] == ["context-quality", "audit-platform", "release-ops"]
assert report["approval_chain"] == ["rfc-4172", "sec-1180", "ops-9301"]
assert report["runbook_steps"] == [
    "freeze-schema",
    "notify-context-quality",
    "verify-marker-retention",
    "publish-war-room-note",
]
assert report["watchlist"] == [
    "legacy-policy-loader",
    "summary-shadow-renderer",
    "owner-ack-cron",
]
assert report["notification_channels"] == ["#audit-war-room", "#context-quality"]
assert report["deadline_matrix"] == {
    "owner_ack": "2026-08-14T09:00Z",
    "schema_freeze": "2026-08-15T12:00Z",
    "release_gate": "2026-08-16T18:00Z",
}
assert report["blocking_conditions"] == [
    "missing-owner-ack",
    "marker-mismatch",
    "policy-drift",
]
assert report["release_flags"] == {
    "requires_manual_review": "yes",
    "allow_legacy_policy": "no",
}
assert report["handoff_ticket"] == "HND-9173-W33"
assert report["review_signoffs"] == ["sig-context-9173", "sig-release-42", "sig-audit-771"]
assert report["rollback_guards"] == [
    "deny-legacy-policy",
    "block-shadow-render",
    "freeze-marker",
]
assert report["control_tags"] == ["sox-audit", "retention-critical", "post-lock"]
""",
    )
    if runtime_checks.returncode == 0:
        required_hits.append("build_report returns exact final contract")
    else:
        missing_required.append("build_report returns exact final contract")

    return _finish_grade(
        target,
        required_hits=required_hits,
        missing_required=missing_required,
        forbidden_hits=forbidden_hits,
        run_tests=run_tests,
        runtime_stdout=runtime_checks.stdout,
        runtime_stderr=runtime_checks.stderr,
    )


def _grade_trace_debug(target: Path, *, run_tests: bool) -> GradeResult:
    decision_path = target / "reconciler" / "decision.py"
    rules_path = target / "reconciler" / "rules.py"
    decision_text = _read_optional(decision_path)
    rules_text = _read_optional(rules_path)
    combined_text = "\n".join([decision_text, rules_text])

    required_hits: list[str] = []
    missing_required: list[str] = []
    forbidden_hits: list[str] = []

    _require_path(decision_path, "reconciler/decision.py exists", required_hits, missing_required)
    _require_path(rules_path, "reconciler/rules.py exists", required_hits, missing_required)
    _require_text(
        decision_text,
        "def build_reconciliation_plan",
        "build_reconciliation_plan is defined",
        required_hits,
        missing_required,
    )
    _require_text(rules_text, "PLAN_KEYS", "PLAN_KEYS schema is preserved", required_hits, missing_required)
    _require_values(combined_text, TRACE_EXPECTED_VALUES, required_hits, missing_required)
    _reject_values(combined_text, TRACE_FORBIDDEN_VALUES, forbidden_hits)

    runtime_checks = _run_python_checks(
        target,
        """
from reconciler.decision import build_reconciliation_plan


def assert_active_plan_matches(plan, expected):
    for key, value in expected.items():
        assert plan[key] == value
    if "default_status_code" in plan:
        assert plan["default_status_code"] == ""
    if "default_owner_chain" in plan:
        assert plan["default_owner_chain"] == []


confirmed = build_reconciliation_plan({
    "tenant": "aurora-ledger",
    "pipeline": "delta-sync",
    "trace_id": "TRACE-KEEP-4821",
    "error_code": "E-RETRY-9173",
    "attempts": "3",
})
assert_active_plan_matches(confirmed, {
    "action": "open-manual-review",
    "owner": "ledger-quality",
    "severity": "sev2",
    "retry_after_minutes": "0",
    "status": "blocked-on-ledger-review",
    "marker": "TRACE-KEEP-4821",
    "queue": "ledger-manual-review",
    "runbook": "rb-ledger-9173",
    "escalation_channel": "#ledger-quality",
    "sla": "PT0M",
    "evidence_hash": "sha256:trace-keep-4821",
    "decision_flags": ["manual-review", "retain-marker", "block-ledger"],
    "rule_id": "rule-ledger-9173",
    "resolver_group": "ledger-quality/oncall",
    "audit_tags": ["manual-review", "retry-threshold", "release-2026.08"],
    "suppressions": ["auto-retry", "candidate-replay"],
})

quarantine = build_reconciliation_plan({
    "tenant": "aurora-ledger",
    "pipeline": "delta-sync",
    "trace_id": "TRACE-QUARANTINE-7712",
    "error_code": "E-CHECKSUM-7712",
    "attempts": "1",
})
assert_active_plan_matches(quarantine, {
    "action": "quarantine-ledger-batch",
    "owner": "ledger-integrity",
    "severity": "sev1",
    "retry_after_minutes": "0",
    "status": "blocked-on-integrity-check",
    "marker": "TRACE-QUARANTINE-7712",
    "queue": "ledger-quarantine",
    "runbook": "rb-ledger-7712",
    "escalation_channel": "#ledger-integrity",
    "sla": "PT0M",
    "evidence_hash": "sha256:trace-quarantine-7712",
    "decision_flags": ["quarantine", "checksum", "block-ledger"],
    "rule_id": "rule-ledger-7712",
    "resolver_group": "ledger-integrity/oncall",
    "audit_tags": ["quarantine", "checksum", "release-2026.08"],
    "suppressions": ["partial-replay", "candidate-quarantine"],
})

audit = build_reconciliation_plan({
    "tenant": "aurora-ledger",
    "pipeline": "audit-sync",
    "trace_id": "TRACE-AUDIT-3345",
    "error_code": "E-AUDIT-LAG-3345",
    "attempts": "2",
})
assert_active_plan_matches(audit, {
    "action": "defer-audit-sync",
    "owner": "audit-quality",
    "severity": "sev3",
    "retry_after_minutes": "30",
    "status": "deferred-for-audit-window",
    "marker": "TRACE-AUDIT-3345",
    "queue": "audit-defer",
    "runbook": "rb-audit-3345",
    "escalation_channel": "#audit-quality",
    "sla": "PT30M",
    "evidence_hash": "sha256:trace-audit-3345",
    "decision_flags": ["audit-defer", "windowed", "retain-marker"],
    "rule_id": "rule-audit-3345",
    "resolver_group": "audit-quality/oncall",
    "audit_tags": ["audit-defer", "windowed", "release-2026.08"],
    "suppressions": ["legacy-audit-retry", "candidate-audit"],
})

other = build_reconciliation_plan({
    "tenant": "other",
    "pipeline": "delta-sync",
    "trace_id": "TRACE-OTHER",
    "error_code": "E-RETRY-9173",
    "attempts": "3",
})
assert other["marker"] == "TRACE-OTHER"
assert other["action"] != "open-manual-review"
assert other["rule_id"] == "rule-default-monitor"
assert other["default_status_code"] == "NO_ACTIVE_RULE"
assert other["default_owner_chain"] == ["triage", "ledger-watch"]
""",
    )
    if runtime_checks.returncode == 0:
        required_hits.append("build_reconciliation_plan applies exact active production rule")
    else:
        missing_required.append("build_reconciliation_plan applies exact active production rule")

    return _finish_grade(
        target,
        required_hits=required_hits,
        missing_required=missing_required,
        forbidden_hits=forbidden_hits,
        run_tests=run_tests,
        runtime_stdout=runtime_checks.stdout,
        runtime_stderr=runtime_checks.stderr,
    )


def _finish_grade(
    target: Path,
    *,
    required_hits: list[str],
    missing_required: list[str],
    forbidden_hits: list[str],
    run_tests: bool,
    runtime_stdout: str,
    runtime_stderr: str,
) -> GradeResult:
    test_exit_code: int | None = None
    test_stdout = runtime_stdout
    test_stderr = runtime_stderr
    if run_tests:
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
            cwd=target,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
        test_exit_code = completed.returncode
        test_stdout += completed.stdout
        test_stderr += completed.stderr
        if completed.returncode != 0:
            forbidden_hits.append("unit tests failed")

    score = 100
    score -= 10 * len(missing_required)
    score -= 15 * len(forbidden_hits)
    score = max(score, 0)
    passed = not missing_required and not forbidden_hits and (test_exit_code in (0, None))
    return GradeResult(
        passed=passed,
        score=score,
        required_hits=required_hits,
        missing_required=missing_required,
        forbidden_hits=forbidden_hits,
        test_exit_code=test_exit_code,
        test_stdout=test_stdout,
        test_stderr=test_stderr,
    )


def _require_path(
    path: Path,
    label: str,
    required_hits: list[str],
    missing_required: list[str],
) -> None:
    if path.is_file():
        required_hits.append(label)
    else:
        missing_required.append(label)


def _require_text(
    text: str,
    needle: str,
    label: str,
    required_hits: list[str],
    missing_required: list[str],
) -> None:
    if needle in text:
        required_hits.append(label)
    else:
        missing_required.append(label)


def _require_values(
    text: str,
    expected_values: dict[str, str],
    required_hits: list[str],
    missing_required: list[str],
) -> None:
    for value, label in expected_values.items():
        if value in text:
            required_hits.append(f"preserves {label}: {value}")
        else:
            missing_required.append(f"preserves {label}: {value}")


def _reject_values(text: str, forbidden_values: Sequence[str], forbidden_hits: list[str]) -> None:
    for forbidden in forbidden_values:
        if forbidden in text:
            forbidden_hits.append(f"uses obsolete value: {forbidden}")


def _run_python_checks(target: Path, code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=target,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )


def _read_case_id(target: Path) -> str | None:
    manifest_path = target / ".ai_job_eval_case.json"
    if not manifest_path.is_file():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    raw_case_id = data.get("case_id")
    return raw_case_id if isinstance(raw_case_id, str) else None


def _read_optional(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grade compress_tool pressure eval target.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--no-tests", action="store_true")
    args = parser.parse_args(argv)

    result = grade_target(Path(args.target), case_id=args.case_id, run_tests=not args.no_tests)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
