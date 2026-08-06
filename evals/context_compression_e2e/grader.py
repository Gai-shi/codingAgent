"""Static + behavioral grader for the context compression E2E fixture."""

from __future__ import annotations

import argparse
import ast
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

    bootstrap_path = target / "sentinel_lab" / "bootstrap.py"
    legacy_path = target / "sentinel_lab" / "legacy_registry.py"

    diff_tool_path, diff_tool_text = _find_diff_review_tool(target)
    bootstrap_text = _read_optional(bootstrap_path)
    implementation_text = "\n".join([diff_tool_text, bootstrap_text])
    diff_tool_tree = _parse_optional(diff_tool_text)
    diff_tool_class = _find_class(diff_tool_tree, "DiffReviewTool")

    _require(diff_tool_path is not None, "DiffReviewTool implementation file exists", required_hits, missing_required)
    _require(
        diff_tool_class is not None and _class_inherits(diff_tool_class, "SentinelToolBase"),
        "DiffReviewTool inherits SentinelToolBase",
        required_hits,
        missing_required,
    )
    _require(_references_name(diff_tool_tree, "GuardedToolOutcome"), "DiffReviewTool references GuardedToolOutcome", required_hits, missing_required)
    _require(_references_name(diff_tool_tree, "MarchConfig"), "DiffReviewTool references MarchConfig", required_hits, missing_required)
    _require(
        diff_tool_class is not None and _class_string_attr(diff_tool_class, "name") == "diff_review",
        'DiffReviewTool declares name = "diff_review"',
        required_hits,
        missing_required,
    )
    _require(
        diff_tool_class is not None
        and _class_string_attr(diff_tool_class, "CONTEXT_RETENTION_MARKER") == "MARCH-CONTEXT-7429",
        "DiffReviewTool preserves early context retention marker",
        required_hits,
        missing_required,
    )
    _require(
        diff_tool_class is not None
        and _class_string_attr(diff_tool_class, "CONFIG_RETENTION_MARKER") == "MARCH-CONFIG-5812",
        "DiffReviewTool preserves config override retention marker",
        required_hits,
        missing_required,
    )
    _require(
        diff_tool_class is not None and _execute_returns_guarded_outcome(diff_tool_class),
        "DiffReviewTool.execute is annotated with GuardedToolOutcome",
        required_hits,
        missing_required,
    )
    _require(_bootstrap_installs_diff_review(bootstrap_text), "bootstrap registers via CommandVault.install", required_hits, missing_required)

    if "legacy_registry" in implementation_text:
        forbidden_hits.append("forbidden import/reference: legacy_registry")
    if "BaseTool" in implementation_text:
        forbidden_hits.append("used obsolete BaseTool contract from noise")
    if "ToolResult" in implementation_text:
        forbidden_hits.append("used obsolete ToolResult contract from noise")
    if "dict[str, Callable]" in implementation_text:
        forbidden_hits.append("forbidden callable registry type: dict[str, Callable]")
    if "register(\"diff_review\"" in implementation_text or "register('diff_review'" in implementation_text:
        forbidden_hits.append("forbidden direct register('diff_review', ...)")
    if (
        "OBSOLETE-MARKER-0000" in diff_tool_text
        or "JSON-LEGACY-1357" in diff_tool_text
        or "CONFIG-JSON-0000" in diff_tool_text
    ):
        forbidden_hits.append("used obsolete context marker from noise")
    raw_return = _find_execute_raw_return(diff_tool_text)
    if raw_return:
        forbidden_hits.append(raw_return)

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


def _find_diff_review_tool(target: Path) -> tuple[Path | None, str]:
    """Find the implementation file that defines DiffReviewTool.

    The benchmark should grade architecture/context retention, not force one
    specific file name. A correct agent may reasonably choose diff_review.py,
    diff_review_tool.py, or another non-legacy module.
    """
    package_dir = target / "sentinel_lab"
    if not package_dir.exists():
        return None, ""

    for path in sorted(package_dir.glob("*.py")):
        if path.name in {"__init__.py", "core.py", "legacy_registry.py"}:
            continue
        text = _read_optional(path)
        if "class DiffReviewTool" in text:
            return path, text
    return None, ""


def _find_execute_raw_return(source: str) -> str | None:
    """Return a diagnostic if DiffReviewTool.execute returns a raw value directly."""
    if not source.strip():
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "DiffReviewTool implementation has syntax error"

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "DiffReviewTool":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "execute":
                    for child in ast.walk(item):
                        if not isinstance(child, ast.Return) or child.value is None:
                            continue
                        value = child.value
                        if isinstance(value, ast.Dict):
                            return "execute returns raw dict"
                        if isinstance(value, ast.List):
                            return "execute returns raw list"
                        if isinstance(value, ast.Tuple):
                            return "execute returns raw tuple"
                        if isinstance(value, ast.Constant) and isinstance(value.value, str):
                            return "execute returns raw str"
                    return None
    return None


def _parse_optional(source: str) -> ast.Module | None:
    if not source.strip():
        return None
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _find_class(tree: ast.Module | None, name: str) -> ast.ClassDef | None:
    if tree is None:
        return None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _class_inherits(node: ast.ClassDef, base_name: str) -> bool:
    return any(_name_of_expr(base) == base_name for base in node.bases)


def _class_string_attr(node: ast.ClassDef, attr_name: str) -> str | None:
    for item in node.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(item, ast.Assign):
            targets = list(item.targets)
            value = item.value
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            targets = [item.target]
            value = item.value
        if value is None:
            continue
        if not any(isinstance(target, ast.Name) and target.id == attr_name for target in targets):
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    return None


def _references_name(tree: ast.Module | None, name: str) -> bool:
    if tree is None:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name:
            return True
        if isinstance(node, ast.Attribute) and node.attr == name:
            return True
    return False


def _execute_returns_guarded_outcome(node: ast.ClassDef) -> bool:
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "execute":
            return _annotation_name(item.returns) == "GuardedToolOutcome"
    return False


def _annotation_name(annotation: ast.expr | None) -> str | None:
    if annotation is None:
        return None
    if isinstance(annotation, ast.Name):
        return annotation.id
    if isinstance(annotation, ast.Attribute):
        return annotation.attr
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return annotation.value
    return None


def _bootstrap_installs_diff_review(source: str) -> bool:
    tree = _parse_optional(source)
    if tree is None:
        return False
    saw_install_call = False
    saw_diff_review = _references_name(tree, "DiffReviewTool")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "install":
            saw_install_call = True
    return saw_install_call and saw_diff_review


def _name_of_expr(expr: ast.expr) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return expr.attr
    return None


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
