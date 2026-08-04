"""Workspace path policy shared by file-oriented tools."""

from __future__ import annotations

from pathlib import Path


DENIED_FILE_NAMES = {".env"}
DENIED_PATH_PARTS = {".ai_job", ".git", ".venv", "__pycache__"}


def resolve_workspace_file(path_text: str, workspace_root: Path) -> Path:
    """Resolve a user/model supplied path and keep it inside workspace_root."""
    if not path_text:
        raise ValueError("missing required argument: path")

    raw_path = Path(path_text).expanduser()
    candidate = raw_path if raw_path.is_absolute() else workspace_root / raw_path
    resolved = candidate.resolve()

    try:
        resolved.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError(f"path escapes workspace: {path_text}") from exc

    relative_parts = resolved.relative_to(workspace_root).parts
    if resolved.name in DENIED_FILE_NAMES or any(part in DENIED_PATH_PARTS for part in relative_parts):
        raise PermissionError(f"refusing to read protected path: {path_text}")

    if not resolved.exists():
        raise FileNotFoundError(f"file not found: {path_text}")
    if not resolved.is_file():
        raise ValueError(f"not a file: {path_text}")

    return resolved


def resolve_workspace_directory(path_text: str, workspace_root: Path) -> Path:
    """Resolve a path as a directory under workspace_root."""
    raw_path = Path(path_text or ".").expanduser()
    candidate = raw_path if raw_path.is_absolute() else workspace_root / raw_path
    resolved = candidate.resolve()

    try:
        resolved.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError(f"path escapes workspace: {path_text}") from exc

    if not resolved.exists():
        raise FileNotFoundError(f"directory not found: {path_text}")
    if not resolved.is_dir():
        raise ValueError(f"not a directory: {path_text}")

    return resolved


def is_hidden_or_protected_dir(path: Path, workspace_root: Path) -> bool:
    relative_parts = path.resolve().relative_to(workspace_root).parts
    return any(part.startswith(".") or part in DENIED_PATH_PARTS for part in relative_parts)


def should_skip_directory(path: Path, allow_hidden_or_protected: bool) -> bool:
    if allow_hidden_or_protected:
        return False
    return path.name.startswith(".") or path.name in DENIED_PATH_PARTS
