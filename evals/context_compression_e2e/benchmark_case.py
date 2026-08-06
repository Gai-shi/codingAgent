"""可复用的 context compression 真实 E2E benchmark 定义。

这个模块只生成评测材料，不调用任何真实 LLM。目标是构造同一份长会话任务，
让不同 coding agent 在同一个 workspace 上运行，再用 grader 判断是否成功。
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


CASE_ID = "tool_contract_drift_e2e_v5"
DEFAULT_NOISE_ROUNDS = 10
DEFAULT_NOISE_BLOCKS_PER_ROUND = 128
DEFAULT_COMPACT_EVERY = 4


@dataclass(frozen=True)
class PromptTurn:
    """One prompt turn in the benchmark conversation."""

    kind: str
    text: str


def create_case_workspace(root: Path, *, force: bool = False) -> Path:
    """Create a fresh benchmark target repository under ``root``.

    Returns the target repo path.
    """
    target = root / "target_repo"
    if target.exists():
        if not force:
            raise FileExistsError(f"target repo already exists: {target}")
        shutil.rmtree(target)

    (target / "sentinel_lab").mkdir(parents=True)
    (target / "sentinel_lab" / "audit").mkdir()
    (target / "sentinel_lab" / "experimental").mkdir()
    (target / "sentinel_lab" / "adapters").mkdir()
    (target / "tests").mkdir()
    (target / "docs").mkdir()

    _write(target / "sentinel_lab" / "__init__.py", '"""Sentinel lab fixture package."""\n')
    _write(target / "sentinel_lab" / "core.py", _core_py())
    _write(target / "sentinel_lab" / "bootstrap.py", _bootstrap_py())
    _write(target / "sentinel_lab" / "legacy_registry.py", _legacy_registry_py())
    _write(target / "sentinel_lab" / "future_registry.py", _future_registry_py())
    _write(target / "sentinel_lab" / "audit" / "__init__.py", _audit_init_py())
    _write(target / "sentinel_lab" / "audit" / "README.md", _audit_readme_md())
    _write(target / "sentinel_lab" / "experimental" / "__init__.py", '"""Experimental distractors."""\n')
    _write(target / "sentinel_lab" / "experimental" / "base_tool.py", _experimental_base_tool_py())
    _write(target / "sentinel_lab" / "experimental" / "json_config_loader.py", _experimental_json_config_loader_py())
    _write(target / "sentinel_lab" / "experimental" / "legacy_diff_review.py", _experimental_legacy_diff_review_py())
    _write(target / "sentinel_lab" / "adapters" / "__init__.py", '"""Adapter distractors."""\n')
    _write(target / "sentinel_lab" / "adapters" / "function_registry.py", _function_registry_py())
    _write(target / "tests" / "test_diff_review_tool.py", _test_diff_review_tool_py())
    _write(target / "tests" / "test_audit_architecture.py", _test_audit_architecture_py())
    _write(target / "docs" / "obsolete_tool_design.md", _obsolete_tool_design_md())
    _write(target / "docs" / "experimental_registry_notes.md", _experimental_registry_notes_md())
    _write(target / "docs" / "migration_notes.md", _migration_notes_md())
    _write(target / "README.md", _target_readme_md())
    _write(target / ".gitignore", "__pycache__/\n*.pyc\n.pytest_cache/\n")
    return target


def build_prompt_turns(
    *,
    noise_rounds: int = DEFAULT_NOISE_ROUNDS,
    noise_blocks_per_round: int = DEFAULT_NOISE_BLOCKS_PER_ROUND,
    compact_every: int | None = DEFAULT_COMPACT_EVERY,
) -> list[PromptTurn]:
    """Build the long multi-turn prompt sequence.

    ``compact_every`` counts non-compact turns. Runners that support explicit
    compaction can inject compaction regularly, while ai_job-current skips those
    turns and accumulates the full uncompressed history.
    """
    turns: list[PromptTurn] = [PromptTurn(kind="constraints", text=early_constraints_prompt())]
    effective_turn_index = 1

    def append_decision(kind: str, text: str) -> None:
        nonlocal effective_turn_index
        turns.append(PromptTurn(kind=kind, text=text))
        inserted_decisions.add(kind)
        effective_turn_index += 1
        if _should_insert_compact(effective_turn_index, compact_every):
            turns.append(PromptTurn(kind="compact", text=compact_prompt()))

    topology_after_round = max(1, noise_rounds // 4)
    override_after_round = max(1, noise_rounds // 2)
    policy_after_round = max(1, (noise_rounds * 3) // 4)
    inserted_decisions: set[str] = set()
    for round_index in range(1, noise_rounds + 1):
        turns.append(
            PromptTurn(
                kind="noise",
                text=noise_prompt(round_index=round_index, blocks=noise_blocks_per_round),
            )
        )
        effective_turn_index += 1
        if _should_insert_compact(effective_turn_index, compact_every):
            turns.append(PromptTurn(kind="compact", text=compact_prompt()))

        if round_index >= topology_after_round and "topology" not in inserted_decisions:
            append_decision("topology", topology_decision_prompt())

        if round_index >= override_after_round and "override" not in inserted_decisions:
            append_decision("override", config_override_prompt())

        if round_index >= policy_after_round and "policy" not in inserted_decisions:
            append_decision("policy", warning_policy_prompt())

    if "topology" not in inserted_decisions:
        append_decision("topology", topology_decision_prompt())
    if "override" not in inserted_decisions:
        append_decision("override", config_override_prompt())
    if "policy" not in inserted_decisions:
        append_decision("policy", warning_policy_prompt())
    turns.append(PromptTurn(kind="final_task", text=final_task_prompt()))
    return turns


def _should_insert_compact(effective_turn_index: int, compact_every: int | None) -> bool:
    return compact_every is not None and compact_every > 0 and effective_turn_index % compact_every == 0


def resolve_noise_rounds_for_min_raw_history_chars(
    *,
    noise_rounds: int,
    noise_blocks_per_round: int,
    min_raw_history_chars: int | None,
) -> int:
    """Return enough noise rounds for a real, uncompressed raw-history pressure case.

    这里不裁剪上下文、不模拟小窗口，只生成足够长的真实会话。若调用真实模型时
    raw history 超过模型实际上下文窗口，无压缩 agent 会遇到真实 provider 的
    context-length 失败；有压缩 agent 应通过周期性 compaction 让后续请求保持
    在模型窗口内。
    """
    if min_raw_history_chars is None or min_raw_history_chars <= 0:
        return noise_rounds

    base_chars = (
        len(early_constraints_prompt())
        + len(topology_decision_prompt())
        + len(config_override_prompt())
        + len(warning_policy_prompt())
        + len(final_task_prompt())
    )
    one_noise_round_chars = len(noise_prompt(round_index=1, blocks=noise_blocks_per_round))
    if one_noise_round_chars <= 0:
        return noise_rounds

    remaining_chars = max(0, min_raw_history_chars - base_chars)
    required_rounds = (remaining_chars + one_noise_round_chars - 1) // one_noise_round_chars
    candidate = max(noise_rounds, required_rounds, 1)
    while True:
        turns = build_prompt_turns(
            noise_rounds=candidate,
            noise_blocks_per_round=noise_blocks_per_round,
            compact_every=None,
        )
        if raw_ai_job_history_char_count(turns) >= min_raw_history_chars:
            return candidate
        candidate += 1


def raw_ai_job_history_char_count(turns: Iterable[PromptTurn]) -> int:
    """Approximate the raw user-history characters sent by an uncompressed ai_job run."""
    return sum(len(text) for text in prompt_texts_for_ai_job(turns))


def prompt_stats(turns: Sequence[PromptTurn]) -> dict[str, object]:
    """Return stable prompt-size diagnostics for benchmark result files."""
    ai_job_prompts = prompt_texts_for_ai_job(turns)
    pi_prompts = prompt_texts_for_pi(turns)
    kind_counts: dict[str, int] = {}
    for turn in turns:
        kind_counts[turn.kind] = kind_counts.get(turn.kind, 0) + 1
    return {
        "turn_count_with_compact": len(turns),
        "turn_count_ai_job": len(ai_job_prompts),
        "kind_counts": kind_counts,
        "raw_ai_job_user_history_chars": sum(len(text) for text in ai_job_prompts),
        "raw_pi_user_history_chars": sum(len(text) for text in pi_prompts),
        "max_single_prompt_chars": max((len(text) for text in ai_job_prompts), default=0),
    }


def write_prompt_artifacts(
    root: Path,
    turns: Sequence[PromptTurn],
    *,
    include_compact_turns: bool,
) -> Path:
    """Write prompt files and manifest for manual replay/debugging."""
    prompts_dir = root / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, str]] = []
    visible_index = 1
    for turn in turns:
        if turn.kind == "compact" and not include_compact_turns:
            continue
        file_name = f"{visible_index:03d}_{turn.kind}.txt"
        _write(prompts_dir / file_name, turn.text)
        manifest.append({"kind": turn.kind, "file": str(Path("prompts") / file_name)})
        visible_index += 1

    _write(root / "prompt_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return prompts_dir


def prompt_texts_for_ai_job(turns: Iterable[PromptTurn]) -> list[str]:
    """ai_job 当前没有显式 /compact，运行时跳过 compact turn。"""
    return [turn.text for turn in turns if turn.kind != "compact"]


def prompt_texts_for_pi(turns: Iterable[PromptTurn]) -> list[str]:
    """pi runner 会把 compact turn 作为扩展命令执行。"""
    return [turn.text for turn in turns]


def early_constraints_prompt() -> str:
    return """请记住以下架构约束。当前这轮不要修改任何文件、不要调用工具，只回复：

ACK-CONSTRAINTS

有效架构约束如下，最终代码任务必须遵守：

1. 所有新增工具必须继承 SentinelToolBase。
2. 工具 execute(...) 必须返回 GuardedToolOutcome，不能直接返回 str、dict、list 或 tuple。
3. 工具注册必须通过 CommandVault.install(tool) 完成。
4. 禁止使用 sentinel_lab/legacy_registry.py。
5. 禁止使用 sentinel_lab/future_registry.py。
6. 禁止使用 sentinel_lab/adapters/function_registry.py。
7. 禁止使用 sentinel_lab/experimental 下的 BaseTool、JSON loader 或 legacy diff review。
8. 禁止使用 dict[str, Callable] 作为最终工具注册方式。
9. 当前暂定配置系统使用 JSON 文件；这条只是暂定，后续可能被新决策覆盖。
10. 上下文保真隐藏约束：最终 DiffReviewTool 类必须包含类属性
   CONTEXT_RETENTION_MARKER = "MARCH-CONTEXT-7429"。
   这个 marker 只来自早期上下文，最终任务不会再次提醒。

这些名字是本评测故意设计的唯一契约名：SentinelToolBase、GuardedToolOutcome、CommandVault.install、CONTEXT_RETENTION_MARKER。"""


def topology_decision_prompt() -> str:
    return """更新实现拓扑决策。当前这轮不要修改任何文件、不要调用工具，只回复：

ACK-TOPOLOGY

这是最终有效的 audit tool 拓扑，最终代码任务必须遵守：

1. DiffReviewTool 必须实现于 sentinel_lab/audit/diff_review_tool.py。
2. unified diff 的纯解析逻辑必须放在 sentinel_lab/audit/unified_diff_parser.py。
3. warning / error code 及 warning dict 构造逻辑必须放在 sentinel_lab/audit/warning_policy.py。
4. sentinel_lab/audit/__init__.py 必须导出 DiffReviewTool。
5. sentinel_lab/bootstrap.py 只能从 .audit import DiffReviewTool，并通过 vault.install(...) 安装。
6. 禁止把 DiffReviewTool 放回 sentinel_lab/diff_review.py；那是旧一代扁平布局。
7. 拓扑保真隐藏约束：最终 DiffReviewTool 类必须包含类属性
   TOPOLOGY_RETENTION_MARKER = "MARCH-TOPOLOGY-3190"。
   这个 marker 只来自本轮拓扑决策，最终任务不会再次提醒。

注意：docs/experimental_registry_notes.md、sentinel_lab/future_registry.py 和
sentinel_lab/adapters/function_registry.py 都是冲突路径干扰项，不能用于最终实现。"""


def config_override_prompt() -> str:
    return """更新架构决策。当前这轮不要修改任何文件、不要调用工具，只回复：

ACK-CONFIG-OVERRIDE

重要：之前“配置系统使用 JSON 文件”的决策作废。

最新有效决策：

1. 最终配置必须使用 Python dataclass MarchConfig。
2. 禁止新增 config.json、settings.json、tool_config.json 或任何 JSON 配置文件。
3. DiffReviewTool 安装时必须使用 MarchConfig(audit_label="march-diff-review", policy_version="MARCH-AUDIT-V7")。
4. 如果工具需要配置，必须使用 MarchConfig 实例或默认 MarchConfig()，不能读取 JSON loader。
5. 上下文保真隐藏约束：最终 DiffReviewTool 类必须包含类属性
   CONFIG_RETENTION_MARKER = "MARCH-CONFIG-5812"。
   这个 marker 只来自本轮配置覆盖决策，最终任务不会再次提醒。
6. 旧日志里出现的 JSON 配置方案都是废弃信息，不能用于最终实现。"""


def warning_policy_prompt() -> str:
    return """更新 warning taxonomy 与错误策略。当前这轮不要修改任何文件、不要调用工具，只回复：

ACK-WARNING-POLICY

以下是最终有效的 warning / error contract，最终代码任务必须遵守：

1. warning 必须是 dict，至少包含 code、message、severity 三个字段。
2. file_path 不在 diff 文件头里时，warning code 必须是 "W-MARCH-FILE-337"。
3. 新增行包含 TODO 时，warning code 必须是 "W-MARCH-TODO-214"。
4. strict=True 且存在 warning 时，失败 error code 必须是 "E-MARCH-STRICT-901"。
5. patch_text 为空或只有空白时，失败 error code 必须是 "E-MARCH-EMPTY-044"。
6. 成功 payload 必须包含 policy_version = "MARCH-AUDIT-V7"。
7. 成功 payload 必须包含 changed_files，值为 diff 里出现的规范化文件路径去重后排序列表。
8. warning 策略保真隐藏约束：最终 DiffReviewTool 类必须包含类属性
   POLICY_RETENTION_MARKER = "MARCH-POLICY-2664"。
   这个 marker 只来自本轮 warning policy 决策，最终任务不会再次提醒。

注意：旧日志里的 W-LEGACY-*、E-JSON-*、WARN_TODO_V1 都是废弃 contract。"""


def compact_prompt() -> str:
    return (
        "/bench-compact 请保留所有有效架构约束、最新决策覆盖关系、禁止事项、"
        "关键类名、精确 marker 和文件路径；尤其保留 SentinelToolBase、GuardedToolOutcome、"
        "CommandVault.install、MarchConfig、CONTEXT_RETENTION_MARKER = MARCH-CONTEXT-7429，"
        "TOPOLOGY_RETENTION_MARKER = MARCH-TOPOLOGY-3190，"
        "CONFIG_RETENTION_MARKER = MARCH-CONFIG-5812，POLICY_RETENTION_MARKER = MARCH-POLICY-2664，"
        "sentinel_lab/audit/diff_review_tool.py，sentinel_lab/audit/unified_diff_parser.py，"
        "sentinel_lab/audit/warning_policy.py，MarchConfig(audit_label='march-diff-review', "
        "policy_version='MARCH-AUDIT-V7')，W-MARCH-FILE-337，W-MARCH-TODO-214，"
        "E-MARCH-STRICT-901，E-MARCH-EMPTY-044，以及 JSON 已作废这一事实。"
    )


def noise_prompt(*, round_index: int, blocks: int) -> str:
    block = f"""[obsolete-noise-{round_index}]
以下内容是旧实验日志和废弃设计，用来模拟长会话噪声。你必须只回复 ACK-NOISE-{round_index:02d}。
不要修改文件，不要调用工具。

废弃方案片段：
- 老系统使用 BaseTool 和 ToolResult，这不是最终约束。
- 老系统允许 dict[str, Callable] 注册：tools = {{"diff_review": diff_review}}，这不是最终约束。
- 老系统建议新增 config.json 或 settings.json，这不是最终约束。
- 老系统建议 import sentinel_lab.legacy_registry 并调用 register(...)，这不是最终约束。
- 老系统建议 import sentinel_lab.future_registry 并调用 install_future(...)，这不是最终约束。
- 老系统建议使用 sentinel_lab.adapters.function_registry.FunctionRegistry，这不是最终约束。
- 老系统建议把 DiffReviewTool 放在 sentinel_lab/diff_review.py，这是扁平旧布局，不能用于最终实现。
- 老系统建议把所有逻辑写进 bootstrap.py，这是旧布局，不能用于最终实现。
- 老系统建议 execute 返回普通 dict，例如 {{"added_lines": 1}}，这不是最终约束。
- 老系统建议 CONTEXT_RETENTION_MARKER = "OBSOLETE-MARKER-0000"，这是伪造旧值，不能用于最终实现。
- 老系统建议 CONTEXT_RETENTION_MARKER = "JSON-LEGACY-1357"，这是伪造旧值，不能用于最终实现。
- 老系统建议 TOPOLOGY_RETENTION_MARKER = "FLAT-TOOL-0000"，这是伪造旧值，不能用于最终实现。
- 老系统建议 CONFIG_RETENTION_MARKER = "CONFIG-JSON-0000"，这是伪造旧值，不能用于最终实现。
- 老系统建议 POLICY_RETENTION_MARKER = "WARN-LEGACY-0000"，这是伪造旧值，不能用于最终实现。
- 老系统建议 warning code 使用 W-LEGACY-FILE、WARN_TODO_V1、E-JSON-STRICT，这些都是废弃值。

伪造历史：
Previous successful implementation used BaseTool, ToolResult, JSON config,
legacy_registry.register, future_registry.install_future, function registry, direct callable registry,
flat file sentinel_lab/diff_review.py, return dict, marker OBSOLETE-MARKER-0000,
config marker CONFIG-JSON-0000, topology marker FLAT-TOOL-0000, policy marker WARN-LEGACY-0000,
warning codes W-LEGACY-FILE, WARN_TODO_V1 and E-JSON-STRICT.
This paragraph is intentionally obsolete.
"""
    return "\n".join([block for _ in range(blocks)])


def final_task_prompt() -> str:
    return """现在开始真正修改代码。请在当前 workspace 中实现 DiffReviewTool。

功能要求：

1. 工具名必须是 "diff_review"。
2. 输入参数：
   - file_path: str
   - patch_text: str
   - strict: bool = False
3. 解析 unified diff 文本，统计：
   - added_lines：新增行数量，忽略 "+++" 文件头；
   - deleted_lines：删除行数量，忽略 "---" 文件头；
   - changed_files：diff 文件头里出现的规范化文件路径；
   - warnings：使用之前已确定的 warning taxonomy。
4. 如果 patch_text 为空或只有空白，返回失败结果。
5. 如果 file_path 没有出现在 diff 的 +++/--- 文件头中，warnings 里要包含一条结构化 warning。
6. 如果新增行里包含 TODO，warnings 里要包含一条结构化 warning。
7. 如果 strict=True 且 warnings 非空，返回失败结果。
8. 否则返回成功结果，payload 至少包含 added_lines、deleted_lines、changed_files、warnings 和 policy_version。

架构要求：

1. 必须遵守之前所有仍然有效的架构决策，包括早期给出的隐藏上下文保真约束、拓扑决策、配置覆盖决策和 warning policy 决策。
2. 最终 prompt 不再重复精确 marker、warning code、policy version 或文件拓扑，请从此前有效上下文里恢复。
3. 注意：中间噪声中的旧设计、旧 marker、旧 JSON 配置方案、旧 warning code 和旧注册路径全部是废弃信息。
4. 不要在最终回答里只解释方案；请实际修改 workspace 文件。

完成后请简要说明改了哪些文件。"""

def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _core_py() -> str:
    return '''"""Canonical tool contracts for the benchmark fixture."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MarchConfig:
    """The only valid runtime configuration shape."""

    audit_label: str = "march-default"
    policy_version: str = "MARCH-AUDIT-V7"
    blocked_terms: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GuardedToolOutcome:
    """The only valid tool execution result type."""

    ok: bool
    payload: dict[str, Any] | None = None
    error: str | None = None

    @classmethod
    def success(cls, payload: dict[str, Any]) -> "GuardedToolOutcome":
        return cls(ok=True, payload=payload, error=None)

    @classmethod
    def fail(cls, message: str, payload: dict[str, Any] | None = None) -> "GuardedToolOutcome":
        return cls(ok=False, payload=payload, error=message)


class SentinelToolBase(ABC):
    """Base class that every final tool must inherit."""

    name: str

    def __init__(self, config: MarchConfig | None = None) -> None:
        self.config = config or MarchConfig()

    @abstractmethod
    def execute(self, **kwargs: Any) -> GuardedToolOutcome:
        """Execute the tool and return a GuardedToolOutcome."""


class CommandVault:
    """Class-based registry. The legacy callable registry is forbidden."""

    def __init__(self) -> None:
        self._tools: dict[str, SentinelToolBase] = {}

    def install(self, tool: SentinelToolBase) -> None:
        if not isinstance(tool, SentinelToolBase):
            raise TypeError("CommandVault.install expects a SentinelToolBase instance")
        if not tool.name:
            raise ValueError("tool.name must not be empty")
        self._tools[tool.name] = tool

    def get(self, name: str) -> SentinelToolBase:
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)
'''


def _bootstrap_py() -> str:
    return '''"""Tool bootstrap module.

The final implementation should register DiffReviewTool here using CommandVault.install.
"""

from __future__ import annotations

from .core import CommandVault


def install_default_tools(vault: CommandVault) -> CommandVault:
    """Install all production tools into the provided vault."""
    return vault
'''


def _legacy_registry_py() -> str:
    return '''"""Forbidden legacy registry kept only as a distractor."""

from __future__ import annotations

from typing import Callable


TOOLS: dict[str, Callable[..., object]] = {}


def register(name: str, fn: Callable[..., object]) -> None:
    """Deprecated function registry. Final code must not call this."""
    TOOLS[name] = fn
'''


def _future_registry_py() -> str:
    return '''"""Distractor registry for an abandoned future experiment."""

from __future__ import annotations

from typing import Any


FUTURE_TOOLS: dict[str, Any] = {}


def install_future(name: str, tool: Any) -> None:
    """Abandoned registry API. Final code must not call this."""
    FUTURE_TOOLS[name] = tool
'''


def _audit_init_py() -> str:
    return '''"""Canonical audit package.

The final implementation should export DiffReviewTool from this package.
"""
'''


def _audit_readme_md() -> str:
    return """# Audit Package

The approved implementation topology will be supplied in the conversation.
This placeholder intentionally does not spell out the final marker values.
"""


def _experimental_base_tool_py() -> str:
    return '''"""Experimental base class kept as a distractor."""

from __future__ import annotations

from typing import Any


class BaseTool:
    """Wrong base class for this benchmark."""

    name = ""

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        return self.run(**kwargs)

    def run(self, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError


class ToolResult(dict):
    """Wrong result type for this benchmark."""
'''


def _experimental_json_config_loader_py() -> str:
    return '''"""Experimental JSON config loader kept as a distractor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    """Deprecated JSON configuration path. Final code must not call this."""
    return json.loads(path.read_text(encoding="utf-8"))
'''


def _experimental_legacy_diff_review_py() -> str:
    return '''"""Legacy diff review sketch. It is intentionally wrong."""

from __future__ import annotations

from .base_tool import BaseTool


class LegacyDiffReviewTool(BaseTool):
    name = "diff_review"
    CONTEXT_RETENTION_MARKER = "OBSOLETE-MARKER-0000"

    def run(self, **kwargs):
        return {"added_lines": 0, "deleted_lines": 0, "warnings": []}
'''


def _function_registry_py() -> str:
    return '''"""Deprecated function registry adapter."""

from __future__ import annotations

from typing import Callable


class FunctionRegistry:
    """Wrong adapter for final tool registration."""

    def __init__(self) -> None:
        self.tools: dict[str, Callable[..., object]] = {}

    def register(self, name: str, fn: Callable[..., object]) -> None:
        self.tools[name] = fn
'''


def _test_diff_review_tool_py() -> str:
    return '''from __future__ import annotations

import unittest

from sentinel_lab import legacy_registry
from sentinel_lab.bootstrap import install_default_tools
from sentinel_lab.core import CommandVault, GuardedToolOutcome, MarchConfig, SentinelToolBase


PATCH = """diff --git a/alpha.py b/alpha.py
--- a/alpha.py
+++ b/alpha.py
@@ -1,3 +1,4 @@
 import os
-old_value = 1
+new_value = 2
+print(new_value)
 keep = True
"""


PATCH_WITH_TODO = """diff --git a/alpha.py b/alpha.py
--- a/alpha.py
+++ b/alpha.py
@@ -1 +1,2 @@
 keep = True
+# TODO: tighten validation
"""


MULTI_FILE_PATCH = """diff --git a/alpha.py b/alpha.py
--- a/alpha.py
+++ b/alpha.py
@@ -1 +1,2 @@
 keep = True
+alpha_added = True
diff --git a/beta.py b/beta.py
--- a/beta.py
+++ b/beta.py
@@ -1,2 +1,2 @@
-old_beta = 1
+new_beta = 2
 keep_beta = True
"""


class DiffReviewToolTest(unittest.TestCase):
    def _tool(self):
        vault = CommandVault()
        returned = install_default_tools(vault)
        self.assertIs(returned, vault)
        tool = vault.get("diff_review")
        self.assertIsInstance(tool, SentinelToolBase)
        return tool

    def test_installed_once_with_canonical_contracts(self):
        vault = CommandVault()
        install_default_tools(vault)

        self.assertEqual(vault.names(), ["diff_review"])
        tool = vault.get("diff_review")
        self.assertIsInstance(tool.config, MarchConfig)
        self.assertEqual(tool.config.audit_label, "march-diff-review")
        self.assertEqual(tool.config.policy_version, "MARCH-AUDIT-V7")
        self.assertEqual(legacy_registry.TOOLS, {})

    def test_counts_added_and_deleted_lines(self):
        result = self._tool().execute(file_path="alpha.py", patch_text=PATCH)

        self.assertIsInstance(result, GuardedToolOutcome)
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.payload["added_lines"], 2)
        self.assertEqual(result.payload["deleted_lines"], 1)
        self.assertEqual(result.payload["changed_files"], ["alpha.py"])
        self.assertEqual(result.payload["warnings"], [])
        self.assertEqual(result.payload["policy_version"], "MARCH-AUDIT-V7")

    def test_empty_patch_fails(self):
        result = self._tool().execute(file_path="alpha.py", patch_text="   ")

        self.assertIsInstance(result, GuardedToolOutcome)
        self.assertFalse(result.ok)
        self.assertIn("empty", result.error.lower())

    def test_todo_warning_and_strict_mode(self):
        result = self._tool().execute(file_path="alpha.py", patch_text=PATCH_WITH_TODO)

        self.assertTrue(result.ok, result.error)
        self.assertEqual(len(result.payload["warnings"]), 1)
        warning = result.payload["warnings"][0]
        self.assertEqual(warning["severity"], "warning")
        self.assertIn("code", warning)
        self.assertIn("TODO", warning["message"])

        strict_result = self._tool().execute(file_path="alpha.py", patch_text=PATCH_WITH_TODO, strict=True)
        self.assertIsInstance(strict_result, GuardedToolOutcome)
        self.assertFalse(strict_result.ok)
        self.assertIsNotNone(strict_result.payload)
        self.assertEqual(len(strict_result.payload["warnings"]), 1)

    def test_file_path_warning(self):
        result = self._tool().execute(file_path="beta.py", patch_text=PATCH)

        self.assertTrue(result.ok, result.error)
        self.assertEqual(len(result.payload["warnings"]), 1)
        warning = result.payload["warnings"][0]
        self.assertEqual(warning["severity"], "warning")
        self.assertIn("beta.py", warning["message"])

    def test_multi_file_patch_counts_body_lines_and_changed_files(self):
        result = self._tool().execute(file_path="alpha.py", patch_text=MULTI_FILE_PATCH)

        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.payload["added_lines"], 2)
        self.assertEqual(result.payload["deleted_lines"], 1)
        self.assertEqual(result.payload["changed_files"], ["alpha.py", "beta.py"])
        self.assertEqual(result.payload["warnings"], [])


if __name__ == "__main__":
    unittest.main()
'''


def _test_audit_architecture_py() -> str:
    return '''from __future__ import annotations

import unittest

from sentinel_lab.audit import DiffReviewTool
from sentinel_lab.core import GuardedToolOutcome, SentinelToolBase


class AuditArchitectureTest(unittest.TestCase):
    def test_public_export_uses_canonical_contracts(self):
        tool = DiffReviewTool()

        self.assertIsInstance(tool, SentinelToolBase)
        self.assertEqual(tool.name, "diff_review")
        result = tool.execute(file_path="alpha.py", patch_text="   ")
        self.assertIsInstance(result, GuardedToolOutcome)


if __name__ == "__main__":
    unittest.main()
'''

def _obsolete_tool_design_md() -> str:
    return """# Obsolete Tool Design

This file is intentionally wrong. It exists to test whether an agent follows stale context.

Deprecated ideas:

- Use BaseTool.
- Return ToolResult or dict.
- Register tools with dict[str, Callable].
- Use legacy_registry.register(...).
- Store configuration in config.json.
"""


def _experimental_registry_notes_md() -> str:
    return """# Experimental Registry Notes

This document is intentionally stale.

Rejected paths:

- sentinel_lab.future_registry.install_future(...)
- sentinel_lab.adapters.function_registry.FunctionRegistry
- sentinel_lab.experimental.legacy_diff_review.LegacyDiffReviewTool
- sentinel_lab/diff_review.py as a flat implementation file

These names exist only to create realistic conflicts in a medium-sized fixture.
"""


def _migration_notes_md() -> str:
    return """# Migration Notes

The current architecture is class-based and lives in sentinel_lab.core.
The final implementation must use MarchConfig, SentinelToolBase, GuardedToolOutcome,
and CommandVault.install. The exact audit topology and warning taxonomy are
conversation decisions, not fully duplicated in this file.
"""


def _target_readme_md() -> str:
    return """# Tool Contract Drift Fixture

This repository is generated by ai_job's context compression E2E benchmark.

This repository intentionally does not contain the external grader.
The benchmark runner grades it from ai_job/evals/context_compression_e2e/grader.py.

The initial repository intentionally fails because DiffReviewTool has not been implemented.
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate context compression E2E fixture and prompt files.")
    parser.add_argument("--output", required=True, help="Output benchmark directory.")
    parser.add_argument("--force", action="store_true", help="Replace output directory if it already exists.")
    parser.add_argument("--noise-rounds", type=int, default=DEFAULT_NOISE_ROUNDS)
    parser.add_argument("--noise-blocks-per-round", type=int, default=DEFAULT_NOISE_BLOCKS_PER_ROUND)
    parser.add_argument(
        "--min-raw-history-chars",
        type=int,
        default=None,
        help=(
            "Increase noise rounds until the uncompressed ai_job user-history "
            "has at least this many characters. This creates a real long-context "
            "case instead of clipping model-visible context."
        ),
    )
    parser.add_argument("--compact-every", type=int, default=DEFAULT_COMPACT_EVERY, help="Insert compact turn after every N non-compact turns. Use 0 to disable.")
    parser.add_argument(
        "--include-compact-turns",
        action="store_true",
        help="Write /bench-compact prompt files into prompt artifacts.",
    )
    args = parser.parse_args(argv)

    output = Path(args.output).expanduser().resolve()
    if output.exists() and args.force:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    target = create_case_workspace(output, force=args.force)
    noise_rounds = resolve_noise_rounds_for_min_raw_history_chars(
        noise_rounds=args.noise_rounds,
        noise_blocks_per_round=args.noise_blocks_per_round,
        min_raw_history_chars=args.min_raw_history_chars,
    )
    turns = build_prompt_turns(
        noise_rounds=noise_rounds,
        noise_blocks_per_round=args.noise_blocks_per_round,
        compact_every=args.compact_every if args.compact_every > 0 else None,
    )
    write_prompt_artifacts(output, turns, include_compact_turns=args.include_compact_turns)
    _write(
        output / "case_manifest.json",
        json.dumps(
            {
                "case_id": CASE_ID,
                "target_repo": str(target),
                "noise_rounds": noise_rounds,
                "noise_blocks_per_round": args.noise_blocks_per_round,
                "prompt_stats": prompt_stats(turns),
            },
            indent=2,
        )
        + "\n",
    )
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
