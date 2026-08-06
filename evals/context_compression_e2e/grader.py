"""Static + behavioral grader for the context compression E2E fixture."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence


FORBIDDEN_JSON_NAMES = {"config.json", "settings.json", "tool_config.json"}


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
    required_hits: list[str] = []
    missing_required: list[str] = []
    forbidden_hits: list[str] = []

    diff_tool_path = target / "sentinel_lab" / "diff_review_tool.py"
    bootstrap_path = target / "sentinel_lab" / "bootstrap.py"
    legacy_path = target / "sentinel_lab" / "legacy_registry.py"

    diff_tool_text = _read_optional(diff_tool_path)
    bootstrap_text = _read_optional(bootstrap_path)
    implementation_text = "\n".join([diff_tool_text, bootstrap_text])

    _require(diff_tool_path.exists(), "sentinel_lab/diff_review_tool.py exists", required_hits, missing_required)
    _require("class DiffReviewTool(SentinelToolBase)" in diff_tool_text, "DiffReviewTool inherits SentinelToolBase", required_hits, missing_required)
    _require("GuardedToolOutcome" in diff_tool_text, "DiffReviewTool references GuardedToolOutcome", required_hits, missing_required)
    _require("MarchConfig" in diff_tool_text, "DiffReviewTool references MarchConfig", required_hits, missing_required)
    _require(".install(" in bootstrap_text and "DiffReviewTool" in bootstrap_text, "bootstrap registers via CommandVault.install", required_hits, missing_required)

    if "legacy_registry" in implementation_text:
        forbidden_hits.append("forbidden import/reference: legacy_registry")
    if "dict[str, Callable]" in implementation_text:
        forbidden_hits.append("forbidden callable registry type: dict[str, Callable]")
    if "register(\"diff_review\"" in implementation_text or "register('diff_review'" in implementation_text:
        forbidden_hits.append("forbidden direct register('diff_review', ...)")
    if "return {" in diff_tool_text or "return dict(" in diff_tool_text:
        forbidden_hits.append("execute appears to return raw dict")
    if "return \"" in diff_tool_text or "return '" in diff_tool_text:
        forbidden_hits.append("execute appears to return raw str")

    legacy_text = _read_optional(legacy_path)
    if "Deprecated function registry" not in legacy_text:
        forbidden_hits.append("sentinel_lab/legacy_registry.py was modified unexpectedly")

    for path in target.rglob("*.json"):
        if path.name in FORBIDDEN_JSON_NAMES:
            forbidden_hits.append(f"forbidden JSON config file: {path.relative_to(target)}")

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
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _require(condition: bool, label: str, hits: list[str], missing: list[str]) -> None:
    if condition:
        hits.append(label)
    else:
        missing.append(label)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grade a generated context compression E2E target repo.")
    parser.add_argument("--target", default=".", help="Target repository to grade.")
    parser.add_argument("--no-tests", action="store_true", help="Skip unittest execution.")
    parser.add_argument("--json", action="store_true", help="Print JSON result only.")
    args = parser.parse_args(argv)

    result = grade_target(Path(args.target), run_tests=not args.no_tests)
    payload = asdict(result)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print("PASS" if result.passed else "FAIL")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
