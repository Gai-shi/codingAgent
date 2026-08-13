"""Grader for the compress_tool pressure eval target repository."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence


EXPECTED_VALUES = {
    "Q4-COMPRESS-AUDIT": "report title",
    "KEEP-COMPRESS-TOOL-9173": "retention marker",
    "POLICY-COMPRESS-42": "policy code",
    "context-quality": "owner",
    "ready-for-review": "status",
    "compress-tool-preserved": "summary prefix",
}
FORBIDDEN_VALUES = (
    "OBSOLETE_MARKER",
    "LEGACY_POLICY",
    "DRAFT_STATUS",
    "wrong-owner",
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


def grade_target(target: Path, *, run_tests: bool = True) -> GradeResult:
    target = target.expanduser().resolve()
    report_path = target / "auditor" / "report.py"
    report_text = _read_optional(report_path)

    required_hits: list[str] = []
    missing_required: list[str] = []
    forbidden_hits: list[str] = []

    if report_path.is_file():
        required_hits.append("auditor/report.py exists")
    else:
        missing_required.append("auditor/report.py exists")

    if "def build_report" in report_text:
        required_hits.append("build_report is defined")
    else:
        missing_required.append("build_report is defined")

    for value, label in EXPECTED_VALUES.items():
        if value in report_text:
            required_hits.append(f"preserves {label}: {value}")
        else:
            missing_required.append(f"preserves {label}: {value}")

    for forbidden in FORBIDDEN_VALUES:
        if forbidden in report_text:
            forbidden_hits.append(f"uses obsolete value: {forbidden}")

    test_exit_code: int | None = None
    test_stdout = ""
    test_stderr = ""
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
        test_stdout = completed.stdout
        test_stderr = completed.stderr
        if completed.returncode != 0:
            forbidden_hits.append("unit tests failed")

    score = 100
    score -= 12 * len(missing_required)
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
    parser.add_argument("--no-tests", action="store_true")
    args = parser.parse_args(argv)

    result = grade_target(Path(args.target), run_tests=not args.no_tests)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
