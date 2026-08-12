"""Default tool assembly."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from .apply_patch_tool import ApplyPatchTool
from .compress_tool import CompressTool
from .grep_tool import GrepTool
from .read_file_tool import ReadFileTool
from .tool_registry import ToolRegistry

if TYPE_CHECKING:
    from ..communication import MessageHistory


def create_default_tool_registry(
    workspace_root: Path,
    request_protected_grep_approval: Optional[Callable[[Path], bool]] = None,
    message_history: "MessageHistory | None" = None,
) -> ToolRegistry:
    tools = [
        ReadFileTool(workspace_root),
        GrepTool(workspace_root, request_protected_grep_approval),
        ApplyPatchTool(workspace_root),
    ]
    if message_history is not None:
        tools.append(CompressTool(message_history))

    return ToolRegistry(
        tools
    )
