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
EXPECTED_DIFF_TOOL_RELATIVE_PATH = Path("sentinel_lab/audit/diff_review_tool.py")
EXPECTED_SUPPORT_FILES = (
    Path("sentinel_lab/audit/unified_diff_parser.py"),
    Path("sentinel_lab/audit/warning_policy.py"),
    Path("sentinel_lab/audit/audit_metadata.py"),
)
EXPECTED_WARNING_CODES = (
    "W-MARCH-FILE-337",
    "W-MARCH-TODO-214",
    "E-MARCH-STRICT-901",
    "E-MARCH-EMPTY-044",
)
EXPECTED_METADATA_CONSTANTS = (
    "MARCH-AUDIT-CHANNEL-42",
    "MARCH-PAYLOAD-SCHEMA-12",
    "march-warning-ledger-17",
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
    required_hits: list[str] = []
    missing_required: list[str] = []
    forbidden_hits: list[str] = []

    bootstrap_path = target / "sentinel_lab" / "bootstrap.py"
    legacy_path = target / "sentinel_lab" / "legacy_registry.py"
    parser_path = target / "sentinel_lab" / "audit" / "unified_diff_parser.py"
    warning_policy_path = target / "sentinel_lab" / "audit" / "warning_policy.py"
    audit_metadata_path = target / "sentinel_lab" / "audit" / "audit_metadata.py"

    diff_tool_path, diff_tool_text = _find_diff_review_tool(target)
    bootstrap_text = _read_optional(bootstrap_path)
    parser_text = _read_optional(parser_path)
    warning_policy_text = _read_optional(warning_policy_path)
    audit_metadata_text = _read_optional(audit_metadata_path)
    implementation_text = _selected_implementation_text(target, diff_tool_path, bootstrap_text)
    diff_tool_tree = _parse_optional(diff_tool_text)
    diff_tool_class = _find_class(diff_tool_tree, "DiffReviewTool")
    parser_tree = _parse_optional(parser_text)

    _require(diff_tool_path is not None, "DiffReviewTool implementation file exists", required_hits, missing_required)
    _require(
        diff_tool_path is not None and diff_tool_path.relative_to(target) == EXPECTED_DIFF_TOOL_RELATIVE_PATH,
        "DiffReviewTool uses approved audit topology path",
        required_hits,
        missing_required,
    )
    for support_file in EXPECTED_SUPPORT_FILES:
        _require((target / support_file).is_file(), f"support file exists: {support_file}", required_hits, missing_required)
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
        diff_tool_class is not None
        and _class_string_attr(diff_tool_class, "TOPOLOGY_RETENTION_MARKER") == "MARCH-TOPOLOGY-3190",
        "DiffReviewTool preserves topology retention marker",
        required_hits,
        missing_required,
    )
    _require(
        diff_tool_class is not None
        and _class_string_attr(diff_tool_class, "POLICY_RETENTION_MARKER") == "MARCH-POLICY-2664",
        "DiffReviewTool preserves warning policy retention marker",
        required_hits,
        missing_required,
    )
    _require(
        diff_tool_class is not None
        and _class_string_attr(diff_tool_class, "METADATA_RETENTION_MARKER") == "MARCH-METADATA-6048",
        "DiffReviewTool preserves metadata retention marker",
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
    _require(
        "MarchConfig" in bootstrap_text
        and "march-diff-review" in bootstrap_text
        and "MARCH-AUDIT-V7" in bootstrap_text,
        "bootstrap installs DiffReviewTool with approved MarchConfig",
        required_hits,
        missing_required,
    )
    _require(
        "from .audit import DiffReviewTool" in bootstrap_text,
        "bootstrap imports DiffReviewTool from audit package",
        required_hits,
        missing_required,
    )
    audit_init_text = _read_optional(target / "sentinel_lab" / "audit" / "__init__.py")
    _require(
        "from .diff_review_tool import DiffReviewTool" in audit_init_text and "__all__" in audit_init_text,
        "audit package exports DiffReviewTool",
        required_hits,
        missing_required,
    )
    for code in EXPECTED_WARNING_CODES:
        _require(code in implementation_text, f"warning policy preserves code: {code}", required_hits, missing_required)
    _require("MARCH-AUDIT-V7" in implementation_text, "payload preserves policy version", required_hits, missing_required)
    _require(
        "PARSER_RETENTION_MARKER" in parser_text and "MARCH-PARSER-7731" in parser_text,
        "parser preserves parser retention marker",
        required_hits,
        missing_required,
    )
    _require(
        _module_has_frozen_dataclass(parser_tree, "UnifiedDiffSummary"),
        "parser defines frozen UnifiedDiffSummary dataclass",
        required_hits,
        missing_required,
    )
    _require(
        _function_return_annotation(parser_tree, "parse_unified_diff") == "UnifiedDiffSummary",
        "parse_unified_diff returns UnifiedDiffSummary",
        required_hits,
        missing_required,
    )
    _require(
        "parse_unified_diff" in diff_tool_text and "UnifiedDiffSummary" in parser_text,
        "DiffReviewTool consumes parser summary contract",
        required_hits,
        missing_required,
    )
    _require(
        "AUDIT_CHANNEL" in audit_metadata_text and "MARCH-AUDIT-CHANNEL-42" in audit_metadata_text,
        "metadata preserves audit channel",
        required_hits,
        missing_required,
    )
    _require(
        "PAYLOAD_SCHEMA_VERSION" in audit_metadata_text and "MARCH-PAYLOAD-SCHEMA-12" in audit_metadata_text,
        "metadata preserves payload schema version",
        required_hits,
        missing_required,
    )
    _require(
        "WARNING_SOURCE" in warning_policy_text and "march-warning-ledger-17" in warning_policy_text,
        "warning policy preserves warning source",
        required_hits,
        missing_required,
    )
    for constant in EXPECTED_METADATA_CONSTANTS:
        _require(
            constant in implementation_text,
            f"implementation preserves metadata constant: {constant}",
            required_hits,
            missing_required,
        )
    for payload_key in ("audit_channel", "payload_schema_version", "audit_label"):
        _require(
            payload_key in diff_tool_text,
            f"payload includes {payload_key}",
            required_hits,
            missing_required,
        )

    if "legacy_registry" in implementation_text:
        forbidden_hits.append("forbidden import/reference: legacy_registry")
    if "future_registry" in implementation_text:
        forbidden_hits.append("forbidden import/reference: future_registry")
    if "FunctionRegistry" in implementation_text or "function_registry" in implementation_text:
        forbidden_hits.append("forbidden function registry adapter")
    if "experimental" in implementation_text:
        forbidden_hits.append("forbidden experimental package reference")
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
        or "FLAT-TOOL-0000" in diff_tool_text
        or "WARN-LEGACY-0000" in diff_tool_text
        or "DICT-PARSER-0000" in implementation_text
        or "LEGACY-META-0000" in implementation_text
    ):
        forbidden_hits.append("used obsolete context marker from noise")
    if "W-LEGACY-FILE" in implementation_text or "WARN_TODO_V1" in implementation_text or "E-JSON-STRICT" in implementation_text:
        forbidden_hits.append("used obsolete warning code from noise")
    if "legacy-audit" in implementation_text or "JSON-PAYLOAD-V1" in implementation_text or "legacy-warning-ledger" in implementation_text:
        forbidden_hits.append("used obsolete metadata constant from noise")
    if "line_counter.py" in implementation_text or (target / "sentinel_lab" / "audit" / "line_counter.py").exists():
        forbidden_hits.append("used obsolete line_counter parser path")
    if _function_return_annotation(parser_tree, "parse_unified_diff") == "dict" or "-> dict" in parser_text:
        forbidden_hits.append("parse_unified_diff returns raw dict instead of UnifiedDiffSummary")
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

    The benchmark now requires the approved audit topology, but this function
    still searches recursively so the grader can report both "class exists" and
    "wrong path" diagnostics.
    """
    package_dir = target / "sentinel_lab"
    if not package_dir.exists():
        return None, ""

    for path in sorted(package_dir.rglob("*.py")):
        relative = path.relative_to(package_dir)
        if relative.parts[0] in {"experimental", "adapters"}:
            continue
        if path.name in {"__init__.py", "core.py", "legacy_registry.py", "future_registry.py"}:
            continue
        text = _read_optional(path)
        if "class DiffReviewTool" in text:
            return path, text
    return None, ""


def _selected_implementation_text(target: Path, diff_tool_path: Path | None, bootstrap_text: str) -> str:
    """Return text from files that are part of the candidate final implementation."""
    chunks = [bootstrap_text]
    audit_dir = target / "sentinel_lab" / "audit"
    if audit_dir.exists():
        for path in sorted(audit_dir.rglob("*.py")):
            chunks.append(_read_optional(path))
    if diff_tool_path is not None and audit_dir not in diff_tool_path.parents:
        chunks.append(_read_optional(diff_tool_path))
    return "\n".join(chunks)


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


def _module_has_frozen_dataclass(tree: ast.Module | None, class_name: str) -> bool:
    node = _find_class(tree, class_name)
    if node is None:
        return False
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        if _name_of_expr(decorator.func) != "dataclass":
            continue
        for keyword in decorator.keywords:
            if keyword.arg == "frozen" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                return True
    return False


def _function_return_annotation(tree: ast.Module | None, function_name: str) -> str | None:
    if tree is None:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return _annotation_name(node.returns)
    return None


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
