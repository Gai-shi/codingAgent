"""Deterministic graders for the runner comparison E2E eval."""

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
    from benchmark_case import (
        CASE_DIRECT_ANSWER,
        CASE_GREP_THEN_READ,
        CASE_MULTI_FILE_PATCH,
        CASE_READ_FILE_ANSWER,
        CASE_SINGLE_FILE_PATCH,
        CASE_TOOL_ERROR_RECOVERY,
    )
else:
    from .benchmark_case import (
        CASE_DIRECT_ANSWER,
        CASE_GREP_THEN_READ,
        CASE_MULTI_FILE_PATCH,
        CASE_READ_FILE_ANSWER,
        CASE_SINGLE_FILE_PATCH,
        CASE_TOOL_ERROR_RECOVERY,
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
    case_id: str | None = None,
    final_answer: str = "",
    run_tests: bool = True,
) -> GradeResult:
    resolved_target = target.expanduser().resolve()
    resolved_case_id = case_id or _read_case_id(resolved_target)
    if resolved_case_id == CASE_DIRECT_ANSWER:
        return _grade_answer(final_answer, required_values=["RUNNER-EVAL-042"], forbidden_values=[])
    if resolved_case_id == CASE_READ_FILE_ANSWER:
        return _grade_answer(
            final_answer,
            required_values=["RF-2026-LG-NATIVE", "eval-platform"],
            forbidden_values=[],
        )
    if resolved_case_id == CASE_GREP_THEN_READ:
        return _grade_answer(
            final_answer,
            required_values=["route-gamma-17", "sha256:runner-gamma-4471"],
            forbidden_values=["retired-route"],
        )
    if resolved_case_id == CASE_SINGLE_FILE_PATCH:
        return _grade_single_file_patch(resolved_target, run_tests=run_tests)
    if resolved_case_id == CASE_MULTI_FILE_PATCH:
        return _grade_multi_file_patch(resolved_target, run_tests=run_tests)
    if resolved_case_id == CASE_TOOL_ERROR_RECOVERY:
        return _grade_answer(
            final_answer,
            required_values=["RC-88-STABLE", "search-first"],
            forbidden_values=[],
        )
    raise ValueError(f"unknown case_id: {resolved_case_id}")


def _grade_answer(
    final_answer: str,
    *,
    required_values: Sequence[str],
    forbidden_values: Sequence[str],
) -> GradeResult:
    required_hits: list[str] = []
    missing_required: list[str] = []
    forbidden_hits: list[str] = []
    for value in required_values:
        if value in final_answer:
            required_hits.append(f"answer includes {value}")
        else:
            missing_required.append(f"answer includes {value}")
    for value in forbidden_values:
        if value in final_answer:
            forbidden_hits.append(f"answer includes forbidden value {value}")

    return _finish_grade(
        required_hits=required_hits,
        missing_required=missing_required,
        forbidden_hits=forbidden_hits,
    )


def _grade_single_file_patch(target: Path, *, run_tests: bool) -> GradeResult:
    calculator_text = _read_optional(target / "calculator.py")
    required_hits: list[str] = []
    missing_required: list[str] = []
    forbidden_hits: list[str] = []

    _require((target / "calculator.py").is_file(), "calculator.py exists", required_hits, missing_required)
    _reject("amount + (amount * percent / 100)" in calculator_text, "keeps broken addition formula", forbidden_hits)

    runtime = _empty_completed_process()
    if run_tests:
        runtime = _run_python_checks(
            target,
            (
                "from calculator import apply_discount\n"
                "assert apply_discount(200, 15) == 170.0\n"
                "assert apply_discount(99, 0) == 99.0\n"
                "assert apply_discount(80, 25) == 60.0\n"
            ),
        )
        if runtime.returncode == 0:
            required_hits.append("runtime discount checks pass")
        else:
            missing_required.append("runtime discount checks pass")

    return _finish_grade(
        required_hits=required_hits,
        missing_required=missing_required,
        forbidden_hits=forbidden_hits,
        test_exit_code=runtime.returncode if run_tests else None,
        test_stdout=runtime.stdout,
        test_stderr=runtime.stderr,
    )


def _grade_multi_file_patch(target: Path, *, run_tests: bool) -> GradeResult:
    status_text = _read_optional(target / "order_status.py")
    formatter_text = _read_optional(target / "formatter.py")
    required_hits: list[str] = []
    missing_required: list[str] = []
    forbidden_hits: list[str] = []

    _require((target / "order_status.py").is_file(), "order_status.py exists", required_hits, missing_required)
    _require((target / "formatter.py").is_file(), "formatter.py exists", required_hits, missing_required)
    _require("Shipped to customer" in status_text, "adds shipped status label", required_hits, missing_required)
    _require("label_for_status" in formatter_text, "formatter uses status label helper", required_hits, missing_required)
    _reject(
        'return f"{order[\'id\']},{order[\'status\']}"' in formatter_text,
        "formatter keeps exact raw status export",
        forbidden_hits,
    )

    runtime = _empty_completed_process()
    if run_tests:
        runtime = _run_python_checks(
            target,
            (
                "from formatter import export_order\n"
                "from order_status import label_for_status\n"
                "assert label_for_status('shipped') == 'Shipped to customer'\n"
                "assert export_order({'id': 'A-100', 'status': 'shipped'}) == 'A-100,Shipped to customer'\n"
                "assert export_order({'id': 'B-200', 'status': 'pending'}) == 'B-200,Pending review'\n"
            ),
        )
        if runtime.returncode == 0:
            required_hits.append("runtime order export checks pass")
        else:
            missing_required.append("runtime order export checks pass")

    return _finish_grade(
        required_hits=required_hits,
        missing_required=missing_required,
        forbidden_hits=forbidden_hits,
        test_exit_code=runtime.returncode if run_tests else None,
        test_stdout=runtime.stdout,
        test_stderr=runtime.stderr,
    )


def _finish_grade(
    *,
    required_hits: list[str],
    missing_required: list[str],
    forbidden_hits: list[str],
    test_exit_code: int | None = None,
    test_stdout: str = "",
    test_stderr: str = "",
) -> GradeResult:
    passed = not missing_required and not forbidden_hits and (test_exit_code in (None, 0))
    return GradeResult(
        passed=passed,
        score=len(required_hits),
        required_hits=required_hits,
        missing_required=missing_required,
        forbidden_hits=forbidden_hits,
        test_exit_code=test_exit_code,
        test_stdout=test_stdout,
        test_stderr=test_stderr,
    )


def _require(condition: bool, label: str, required_hits: list[str], missing_required: list[str]) -> None:
    if condition:
        required_hits.append(label)
    else:
        missing_required.append(label)


def _reject(condition: bool, label: str, forbidden_hits: list[str]) -> None:
    if condition:
        forbidden_hits.append(label)


def _read_case_id(target: Path) -> str:
    manifest_path = target / ".ai_job_eval_case.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    case_id = data.get("case_id")
    if not isinstance(case_id, str):
        raise ValueError(f"missing case_id in {manifest_path}")
    return case_id


def _read_optional(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _run_python_checks(target: Path, code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=target,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )


def _empty_completed_process() -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grade a runner comparison eval target.")
    parser.add_argument("target")
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--final-answer", default="")
    parser.add_argument("--no-tests", action="store_true")
    args = parser.parse_args(argv)

    grade = grade_target(
        Path(args.target),
        case_id=args.case_id,
        final_answer=args.final_answer,
        run_tests=not args.no_tests,
    )
    print(json.dumps(asdict(grade), ensure_ascii=False, indent=2))
    return 0 if grade.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
