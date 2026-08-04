"""apply_patch tool."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .base_tool import BaseTool
from .git_diff_apply import FileApplyResult, GitDiffApplyError, apply_file_patch_to_text
from .git_diff_parser import FilePatch, GitDiffParseError, GitDiffPatch, parse_git_diff
from .path_policy import resolve_workspace_patch_path


APPLY_PATCH_DESCRIPTION = (
    "Apply a git diff patch inside the current workspace. Supports modifying, adding, "
    "and deleting UTF-8 text files. All files are prechecked before anything is written."
)
APPLY_PATCH_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "patch": {
            "type": "string",
            "description": "A git diff patch to apply. Use diff --git headers and unified hunks.",
        }
    },
    "required": ["patch"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class PlannedFileChange:
    relative_path: str
    absolute_path: Path
    result: FileApplyResult
    hunk_count: int


def apply_patch_text(arguments: dict[str, Any], workspace_root: Path) -> str:
    patch_value = arguments.get("patch")
    if not isinstance(patch_value, str):
        raise ValueError('invalid arguments: "patch" must be a string')

    parsed = parse_git_diff(patch_value)
    planned_changes = _plan_patch_application(parsed, workspace_root)
    _write_planned_changes(planned_changes)
    return _format_success(planned_changes)


class ApplyPatchTool(BaseTool):
    name = "apply_patch"
    description = APPLY_PATCH_DESCRIPTION
    parameters_schema = APPLY_PATCH_PARAMETERS_SCHEMA

    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = workspace_root

    def _run(self, arguments: dict[str, Any]) -> str:
        return apply_patch_text(arguments, self._workspace_root)


def _plan_patch_application(parsed: GitDiffPatch, workspace_root: Path) -> list[PlannedFileChange]:
    planned: list[PlannedFileChange] = []
    seen_paths: set[Path] = set()

    for file_patch in parsed.files:
        relative_path = _file_patch_relative_path(file_patch)
        absolute_path = resolve_workspace_patch_path(relative_path, workspace_root, file_patch.operation)
        if absolute_path in seen_paths:
            raise ValueError(f"patch contains multiple file diffs for the same path: {relative_path}")
        seen_paths.add(absolute_path)

        original_text = _read_original_text_for_precheck(file_patch, absolute_path, relative_path)
        result = apply_file_patch_to_text(file_patch, original_text)
        planned.append(
            PlannedFileChange(
                relative_path=relative_path,
                absolute_path=absolute_path,
                result=result,
                hunk_count=len(file_patch.hunks),
            )
        )

    return planned


def _read_original_text_for_precheck(
    file_patch: FilePatch,
    absolute_path: Path,
    relative_path: str,
) -> Optional[str]:
    if file_patch.operation == "add":
        if absolute_path.exists():
            raise FileExistsError(f"new file already exists: {relative_path}")
        return None

    if not absolute_path.exists():
        raise FileNotFoundError(f"file not found for {file_patch.operation}: {relative_path}")
    if not absolute_path.is_file():
        raise ValueError(f"not a file: {relative_path}")

    try:
        return absolute_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"file is not valid UTF-8 text: {relative_path}") from exc


def _write_planned_changes(planned_changes: list[PlannedFileChange]) -> None:
    for planned in planned_changes:
        if planned.result.operation == "delete":
            planned.absolute_path.unlink()
            continue

        if planned.result.content is None:
            raise RuntimeError(f"missing content for {planned.result.operation}: {planned.relative_path}")
        planned.absolute_path.parent.mkdir(parents=True, exist_ok=True)
        planned.absolute_path.write_text(planned.result.content, encoding="utf-8")


def _format_success(planned_changes: list[PlannedFileChange]) -> str:
    hunk_count = sum(change.hunk_count for change in planned_changes)
    lines = [
        f"Successfully applied patch: {len(planned_changes)} file(s) changed, {hunk_count} hunk(s)."
    ]

    for title, operation in (("Modified", "modify"), ("Added", "add"), ("Deleted", "delete")):
        paths = [change.relative_path for change in planned_changes if change.result.operation == operation]
        if not paths:
            continue
        lines.append(f"{title}:")
        lines.extend(f"- {path}" for path in paths)

    return "\n".join(lines)


def _file_patch_relative_path(file_patch: FilePatch) -> str:
    path = file_patch.new_path if file_patch.new_path is not None else file_patch.old_path
    if path is None:
        raise ValueError("file patch has neither old nor new path")
    return path


__all__ = [
    "APPLY_PATCH_DESCRIPTION",
    "APPLY_PATCH_PARAMETERS_SCHEMA",
    "ApplyPatchTool",
    "apply_patch_text",
    "GitDiffApplyError",
    "GitDiffParseError",
]
