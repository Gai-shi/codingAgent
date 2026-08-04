"""In-memory application for parsed git-diff patches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .git_diff_parser import FilePatch, Hunk, HunkLine


class GitDiffApplyError(ValueError):
    """Raised when a parsed patch cannot be applied safely."""


@dataclass(frozen=True)
class HunkReplacement:
    start: int
    end: int
    old_block: str
    new_block: str
    hunk_index: int


@dataclass(frozen=True)
class FileApplyResult:
    path: str
    operation: str
    content: Optional[str]


def apply_file_patch_to_text(file_patch: FilePatch, original_text: Optional[str]) -> FileApplyResult:
    """Apply one FilePatch to text in memory.

    The caller is responsible for filesystem checks such as whether an added
    file already exists. This function only validates and transforms content.
    """
    if file_patch.operation == "add":
        if original_text is not None:
            raise GitDiffApplyError(f"new file already has original content: {file_patch.new_path}")
        return FileApplyResult(
            path=_result_path(file_patch),
            operation="add",
            content=_build_added_file_content(file_patch),
        )

    if original_text is None:
        raise GitDiffApplyError(f"original content is required for {file_patch.operation}: {_result_path(file_patch)}")

    if file_patch.operation == "modify":
        return FileApplyResult(
            path=_result_path(file_patch),
            operation="modify",
            content=_apply_modify_patch(file_patch, original_text),
        )

    if file_patch.operation == "delete":
        next_content = _apply_modify_patch(file_patch, original_text)
        if next_content != "":
            raise GitDiffApplyError(
                f"delete patch for {_result_path(file_patch)} did not remove the entire file"
            )
        return FileApplyResult(path=_result_path(file_patch), operation="delete", content=None)

    raise GitDiffApplyError(f"unsupported file patch operation: {file_patch.operation}")


def _build_added_file_content(file_patch: FilePatch) -> str:
    if file_patch.operation != "add":
        raise GitDiffApplyError("_build_added_file_content expects an add patch")

    chunks: list[str] = []
    for hunk_index, hunk in enumerate(file_patch.hunks, start=1):
        old_block = _hunk_old_block(hunk)
        if old_block != "":
            raise GitDiffApplyError(
                f"add patch for {_result_path(file_patch)} hunk {hunk_index} unexpectedly contains old content"
            )
        chunks.append(_hunk_new_block(hunk))
    return "".join(chunks)


def _apply_modify_patch(file_patch: FilePatch, original_text: str) -> str:
    replacements = _prepare_replacements(file_patch, original_text)
    next_text = original_text
    for replacement in sorted(replacements, key=lambda item: item.start, reverse=True):
        next_text = next_text[: replacement.start] + replacement.new_block + next_text[replacement.end :]
    return next_text


def _prepare_replacements(file_patch: FilePatch, original_text: str) -> list[HunkReplacement]:
    replacements: list[HunkReplacement] = []
    for hunk_index, hunk in enumerate(file_patch.hunks, start=1):
        old_block = _hunk_old_block(hunk)
        new_block = _hunk_new_block(hunk)
        if old_block == "":
            raise GitDiffApplyError(
                f"hunk {hunk_index} in {_result_path(file_patch)} has empty old content and cannot be located safely"
            )

        matches = _find_all_occurrences(original_text, old_block)
        if not matches:
            raise GitDiffApplyError(f"hunk {hunk_index} in {_result_path(file_patch)} did not match any content")
        if len(matches) > 1:
            candidate_lines = ", ".join(str(_line_number_at(original_text, match)) for match in matches)
            raise GitDiffApplyError(
                f"hunk {hunk_index} in {_result_path(file_patch)} matched multiple locations at lines "
                f"{candidate_lines}; add more surrounding context"
            )

        start = matches[0]
        replacements.append(
            HunkReplacement(
                start=start,
                end=start + len(old_block),
                old_block=old_block,
                new_block=new_block,
                hunk_index=hunk_index,
            )
        )

    _validate_non_overlapping(replacements, _result_path(file_patch))
    return replacements


def _validate_non_overlapping(replacements: list[HunkReplacement], path: str) -> None:
    ordered = sorted(replacements, key=lambda item: item.start)
    for previous, current in zip(ordered, ordered[1:]):
        if previous.end > current.start:
            raise GitDiffApplyError(
                f"hunks {previous.hunk_index} and {current.hunk_index} in {path} overlap; merge them into one hunk"
            )


def _hunk_old_block(hunk: Hunk) -> str:
    return _join_side_lines(
        line for line in hunk.lines if line.kind in {"context", "remove"}
    )


def _hunk_new_block(hunk: Hunk) -> str:
    return _join_side_lines(
        line for line in hunk.lines if line.kind in {"context", "add"}
    )


def _join_side_lines(lines) -> str:
    chunks: list[str] = []
    for line in lines:
        chunks.append(line.text)
        if not _line_has_no_newline_for_side(line):
            chunks.append("\n")
    return "".join(chunks)


def _line_has_no_newline_for_side(line: HunkLine) -> bool:
    if line.kind == "context":
        return line.old_no_newline and line.new_no_newline
    if line.kind == "remove":
        return line.old_no_newline
    return line.new_no_newline


def _find_all_occurrences(content: str, needle: str) -> list[int]:
    if needle == "":
        return []
    matches: list[int] = []
    start = 0
    while True:
        index = content.find(needle, start)
        if index == -1:
            return matches
        matches.append(index)
        start = index + 1


def _line_number_at(content: str, index: int) -> int:
    return content.count("\n", 0, index) + 1


def _result_path(file_patch: FilePatch) -> str:
    path = file_patch.new_path if file_patch.new_path is not None else file_patch.old_path
    if path is None:
        raise GitDiffApplyError("file patch has neither old nor new path")
    return path
