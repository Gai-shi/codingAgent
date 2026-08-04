"""Parser for the git-diff subset used by apply_patch."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional


PatchOperation = Literal["modify", "add", "delete"]
HunkLineKind = Literal["context", "remove", "add"]


class GitDiffParseError(ValueError):
    """Raised when a patch is outside the supported git-diff subset."""


@dataclass(frozen=True)
class HunkLine:
    kind: HunkLineKind
    text: str
    old_no_newline: bool = False
    new_no_newline: bool = False


@dataclass(frozen=True)
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    section_header: str = ""
    lines: list[HunkLine] = field(default_factory=list)


@dataclass(frozen=True)
class FilePatch:
    old_path: Optional[str]
    new_path: Optional[str]
    operation: PatchOperation
    hunks: list[Hunk] = field(default_factory=list)


@dataclass(frozen=True)
class GitDiffPatch:
    files: list[FilePatch] = field(default_factory=list)


_DIFF_HEADER_RE = re.compile(r"^diff --git (\S+) (\S+)$")
_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: (.*))?$")
_NO_NEWLINE_MARKER = r"\ No newline at end of file"
_UNSUPPORTED_FILE_HEADERS = (
    "rename from ",
    "rename to ",
    "copy from ",
    "copy to ",
    "similarity index ",
    "dissimilarity index ",
    "GIT binary patch",
    "Binary files ",
)


def parse_git_diff(patch_text: str) -> GitDiffPatch:
    """Parse a strict git-diff subset into structured file patches.

    Supported operations are normal modifications, new files, and deleted files.
    Rename/copy/binary patches and quoted paths are intentionally rejected in
    the first version.
    """
    if not isinstance(patch_text, str) or not patch_text.strip():
        raise GitDiffParseError("patch must be a non-empty string")

    lines = patch_text.splitlines()
    index = 0
    files: list[FilePatch] = []

    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if not line.startswith("diff --git "):
            raise GitDiffParseError(f"expected 'diff --git' header at line {index + 1}")

        file_patch, index = _parse_file_patch(lines, index)
        files.append(file_patch)

    if not files:
        raise GitDiffParseError("patch does not contain any file diffs")
    return GitDiffPatch(files=files)


def _parse_file_patch(lines: list[str], start_index: int) -> tuple[FilePatch, int]:
    header = lines[start_index]
    header_match = _DIFF_HEADER_RE.match(header)
    if not header_match:
        raise GitDiffParseError(
            f"unsupported diff header at line {start_index + 1}: expected simple paths without spaces"
        )
    _require_prefixed_path(header_match.group(1), "a/", start_index + 1)
    _require_prefixed_path(header_match.group(2), "b/", start_index + 1)

    index = start_index + 1
    old_path: Optional[str] = None
    new_path: Optional[str] = None
    hunks: list[Hunk] = []

    while index < len(lines):
        line = lines[index]
        if line.startswith("diff --git "):
            break
        if line.startswith(_UNSUPPORTED_FILE_HEADERS):
            raise GitDiffParseError(f"unsupported git diff header at line {index + 1}: {line}")
        if line.startswith("--- "):
            old_path = _parse_old_path(line, index + 1)
            index += 1
            if index >= len(lines) or not lines[index].startswith("+++ "):
                raise GitDiffParseError(f"expected '+++' path header after line {index}")
            new_path = _parse_new_path(lines[index], index + 1)
            index += 1
            break
        index += 1

    if old_path is None and new_path is None:
        raise GitDiffParseError(f"file diff starting at line {start_index + 1} is missing ---/+++ headers")

    operation = _detect_operation(old_path, new_path)

    while index < len(lines):
        line = lines[index]
        if line.startswith("diff --git "):
            break
        if line.startswith(_UNSUPPORTED_FILE_HEADERS):
            raise GitDiffParseError(f"unsupported git diff header at line {index + 1}: {line}")
        if not line:
            index += 1
            continue
        if not line.startswith("@@ "):
            raise GitDiffParseError(f"expected hunk header at line {index + 1}")
        hunk, index = _parse_hunk(lines, index)
        hunks.append(hunk)

    if not hunks:
        raise GitDiffParseError(f"file diff starting at line {start_index + 1} does not contain any hunks")

    return FilePatch(old_path=old_path, new_path=new_path, operation=operation, hunks=hunks), index


def _parse_hunk(lines: list[str], start_index: int) -> tuple[Hunk, int]:
    header = lines[start_index]
    match = _HUNK_HEADER_RE.match(header)
    if not match:
        raise GitDiffParseError(f"invalid hunk header at line {start_index + 1}")

    old_start = int(match.group(1))
    old_count = int(match.group(2) or "1")
    new_start = int(match.group(3))
    new_count = int(match.group(4) or "1")
    section_header = match.group(5) or ""
    hunk_lines: list[HunkLine] = []

    index = start_index + 1
    while index < len(lines):
        line = lines[index]
        if line.startswith("diff --git ") or line.startswith("@@ "):
            break
        if line == _NO_NEWLINE_MARKER:
            if not hunk_lines:
                raise GitDiffParseError(f"no-newline marker at line {index + 1} has no preceding hunk line")
            hunk_lines[-1] = _mark_no_newline(hunk_lines[-1])
            index += 1
            continue
        if not line:
            raise GitDiffParseError(f"empty patch line at line {index + 1} is missing a hunk prefix")

        prefix = line[0]
        text = line[1:]
        if prefix == " ":
            hunk_lines.append(HunkLine(kind="context", text=text))
        elif prefix == "-":
            hunk_lines.append(HunkLine(kind="remove", text=text))
        elif prefix == "+":
            hunk_lines.append(HunkLine(kind="add", text=text))
        elif line.startswith("--- ") or line.startswith("+++ "):
            raise GitDiffParseError(f"unexpected file header inside hunk at line {index + 1}")
        else:
            raise GitDiffParseError(f"invalid hunk line prefix at line {index + 1}: {prefix!r}")
        index += 1

    _validate_hunk_counts(hunk_lines, old_count, new_count, start_index + 1)
    return (
        Hunk(
            old_start=old_start,
            old_count=old_count,
            new_start=new_start,
            new_count=new_count,
            section_header=section_header,
            lines=hunk_lines,
        ),
        index,
    )


def _mark_no_newline(line: HunkLine) -> HunkLine:
    if line.kind == "context":
        return HunkLine(kind=line.kind, text=line.text, old_no_newline=True, new_no_newline=True)
    if line.kind == "remove":
        return HunkLine(kind=line.kind, text=line.text, old_no_newline=True, new_no_newline=False)
    return HunkLine(kind=line.kind, text=line.text, old_no_newline=False, new_no_newline=True)


def _validate_hunk_counts(lines: list[HunkLine], old_count: int, new_count: int, hunk_line_number: int) -> None:
    actual_old_count = sum(1 for line in lines if line.kind in {"context", "remove"})
    actual_new_count = sum(1 for line in lines if line.kind in {"context", "add"})
    if actual_old_count != old_count:
        raise GitDiffParseError(
            f"hunk at line {hunk_line_number} declares {old_count} old lines but contains {actual_old_count}"
        )
    if actual_new_count != new_count:
        raise GitDiffParseError(
            f"hunk at line {hunk_line_number} declares {new_count} new lines but contains {actual_new_count}"
        )


def _parse_old_path(line: str, line_number: int) -> Optional[str]:
    value = line[4:].strip()
    if value == "/dev/null":
        return None
    return _strip_git_prefix(value, "a/", line_number)


def _parse_new_path(line: str, line_number: int) -> Optional[str]:
    value = line[4:].strip()
    if value == "/dev/null":
        return None
    return _strip_git_prefix(value, "b/", line_number)


def _strip_git_prefix(path: str, expected_prefix: str, line_number: int) -> str:
    _require_prefixed_path(path, expected_prefix, line_number)
    stripped = path[len(expected_prefix) :]
    if not stripped:
        raise GitDiffParseError(f"empty path at line {line_number}")
    return stripped


def _require_prefixed_path(path: str, expected_prefix: str, line_number: int) -> None:
    if path.startswith('"'):
        raise GitDiffParseError(f"quoted paths are not supported at line {line_number}")
    if not path.startswith(expected_prefix):
        raise GitDiffParseError(f"expected path with {expected_prefix!r} prefix at line {line_number}")


def _detect_operation(old_path: Optional[str], new_path: Optional[str]) -> PatchOperation:
    if old_path is None and new_path is not None:
        return "add"
    if old_path is not None and new_path is None:
        return "delete"
    if old_path is None and new_path is None:
        raise GitDiffParseError("file patch cannot have both old and new path as /dev/null")
    if old_path != new_path:
        raise GitDiffParseError("renames are not supported in the first apply_patch parser")
    return "modify"
