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
}
CONFLICT_FORBIDDEN_VALUES = (
    "OBSOLETE_MARKER",
    "LEGACY_POLICY",
    "DRAFT_STATUS",
    "wrong-owner",
    "shadow-summary",
)

TRACE_EXPECTED_VALUES = {
    "open-manual-review": "action",
    "ledger-quality": "owner",
    "sev2": "severity",
    "0": "retry delay",
    "blocked-on-ledger-review": "status",
}
TRACE_FORBIDDEN_VALUES = (
    "retry-later",
    "legacy-retry",
    "wrong-owner",
    "TRACE-OBSOLETE",
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
    report_text = _read_optional(report_path)

    required_hits: list[str] = []
    missing_required: list[str] = []
    forbidden_hits: list[str] = []

    _require_path(report_path, "auditor/report.py exists", required_hits, missing_required)
    _require_text(report_text, "def build_report", "build_report is defined", required_hits, missing_required)
    _require_values(report_text, CONFLICT_EXPECTED_VALUES, required_hits, missing_required)
    _reject_values(report_text, CONFLICT_FORBIDDEN_VALUES, forbidden_hits)

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
}
for key, value in expected.items():
    assert report[key] == value
assert report["summary"].startswith("compress-tool-preserved")
assert "KEEP-COMPRESS-TOOL-9173" in report["summary"]
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
    decision_text = _read_optional(decision_path)

    required_hits: list[str] = []
    missing_required: list[str] = []
    forbidden_hits: list[str] = []

    _require_path(decision_path, "reconciler/decision.py exists", required_hits, missing_required)
    _require_text(
        decision_text,
        "def build_reconciliation_plan",
        "build_reconciliation_plan is defined",
        required_hits,
        missing_required,
    )
    _require_values(decision_text, TRACE_EXPECTED_VALUES, required_hits, missing_required)
    _reject_values(decision_text, TRACE_FORBIDDEN_VALUES, forbidden_hits)

    runtime_checks = _run_python_checks(
        target,
        """
from reconciler.decision import build_reconciliation_plan

confirmed = build_reconciliation_plan({
    "tenant": "aurora-ledger",
    "pipeline": "delta-sync",
    "trace_id": "TRACE-KEEP-4821",
    "error_code": "E-RETRY-9173",
    "attempts": "3",
})
assert confirmed == {
    "action": "open-manual-review",
    "owner": "ledger-quality",
    "severity": "sev2",
    "retry_after_minutes": "0",
    "status": "blocked-on-ledger-review",
    "marker": "TRACE-KEEP-4821",
}

other = build_reconciliation_plan({
    "tenant": "other",
    "pipeline": "delta-sync",
    "trace_id": "TRACE-OTHER",
    "error_code": "E-RETRY-9173",
    "attempts": "3",
})
assert other["marker"] == "TRACE-OTHER"
assert other["action"] != "open-manual-review"
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
