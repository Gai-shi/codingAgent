"""Fixture and prompts for the compress_tool real-LLM pressure eval."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


CASE_ID = "compress_tool_pressure_v1"
DEFAULT_NOISE_BLOCKS = 420


@dataclass(frozen=True)
class PromptTurn:
    kind: str
    text: str


def create_case_workspace(root: Path, *, force: bool = False, noise_blocks: int = DEFAULT_NOISE_BLOCKS) -> Path:
    target = root / "target_repo"
    if target.exists():
        if not force:
            raise FileExistsError(f"target repo already exists: {target}")
        shutil.rmtree(target)

    (target / "auditor").mkdir(parents=True)
    (target / "evidence").mkdir()
    (target / "tests").mkdir()

    _write(target / "auditor" / "__init__.py", '"""Audit fixture package."""\n')
    _write(target / "auditor" / "report.py", _report_py())
    _write(target / "tests" / "test_report.py", _test_report_py())
    _write(target / "evidence" / "noisy_audit_log.txt", noisy_evidence_text(noise_blocks=noise_blocks))
    _write(target / "README.md", _target_readme_md())
    _write(target / ".gitignore", "__pycache__/\n*.pyc\n")
    return target


def build_prompt_turns() -> list[PromptTurn]:
    return [PromptTurn(kind="final_task", text=final_task_prompt())]


def write_prompt_artifacts(root: Path, turns: Sequence[PromptTurn]) -> Path:
    prompts_dir = root / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    for index, turn in enumerate(turns, start=1):
        file_name = f"{index:03d}_{turn.kind}.txt"
        _write(prompts_dir / file_name, turn.text)
        manifest.append({"kind": turn.kind, "file": str(Path("prompts") / file_name)})
    _write(root / "prompt_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return prompts_dir


def prompt_texts(turns: Sequence[PromptTurn]) -> list[str]:
    return [turn.text for turn in turns]


def prompt_stats(turns: Sequence[PromptTurn], *, noise_blocks: int) -> dict[str, object]:
    prompts = prompt_texts(turns)
    evidence = noisy_evidence_text(noise_blocks=noise_blocks)
    return {
        "turn_count": len(turns),
        "prompt_chars": sum(len(text) for text in prompts),
        "evidence_chars": len(evidence),
        "noise_blocks": noise_blocks,
    }


def final_task_prompt() -> str:
    return """请完成当前 workspace 里的 auditor 报告实现。

强制流程：

1. 你必须先读取 evidence/noisy_audit_log.txt。
2. 这个 evidence 文件很长，包含大量废弃方案和少量 FINAL_CONTRACT 信息。
3. 读取后，如果你有 compress_tool，请用它把 read_file 的工具输出压缩成只包含 FINAL_CONTRACT、禁止项、文件路径和测试相关事实的摘要。
4. 然后实现 auditor/report.py，让 tests/ 下的单测通过。

任务约束：

- 不要凭空猜测最终常量；必须从 evidence/noisy_audit_log.txt 里恢复。
- 不要使用 evidence 里的 OBSOLETE_*、LEGACY_*、DRAFT_* 信息。
- 当前 ai_job 的 apply_patch 工具只接受标准 git diff 文本，必须以 diff --git 文件头组织 patch。
- 不要把 "*** Begin Patch" / "*** End Patch" 格式传给 apply_patch 工具。

完成后简要说明改了哪些文件。"""


def noisy_evidence_text(*, noise_blocks: int = DEFAULT_NOISE_BLOCKS) -> str:
    blocks: list[str] = [_final_contract_block("early")]
    for index in range(1, noise_blocks + 1):
        blocks.append(_noise_block(index))
        if index == max(1, noise_blocks // 2):
            blocks.append(_final_contract_block("middle"))
    blocks.append(_final_contract_block("late"))
    return "\n\n".join(blocks) + "\n"


def _final_contract_block(label: str) -> str:
    return f"""[FINAL_CONTRACT::{label}]
These are the only valid values for the final implementation:
- REPORT_TITLE = "Q4-COMPRESS-AUDIT"
- REQUIRED_MARKER = "KEEP-COMPRESS-TOOL-9173"
- POLICY_CODE = "POLICY-COMPRESS-42"
- OWNER = "context-quality"
- FINAL_STATUS = "ready-for-review"
- SUMMARY_PREFIX = "compress-tool-preserved"

Implementation requirements:
- auditor/report.py must expose build_report() -> dict[str, str].
- build_report()["title"] must equal REPORT_TITLE.
- build_report()["marker"] must equal REQUIRED_MARKER.
- build_report()["policy_code"] must equal POLICY_CODE.
- build_report()["owner"] must equal OWNER.
- build_report()["status"] must equal FINAL_STATUS.
- build_report()["summary"] must start with SUMMARY_PREFIX.
- The implementation must not include OBSOLETE_MARKER, LEGACY_POLICY, DRAFT_STATUS, or wrong-owner.
"""


def _noise_block(index: int) -> str:
    return f"""[obsolete-block-{index:04d}]
This block is intentionally noisy and obsolete.
Old proposal:
- REPORT_TITLE = "OBSOLETE-AUDIT-{index:04d}"
- REQUIRED_MARKER = "OBSOLETE_MARKER_{index:04d}"
- POLICY_CODE = "LEGACY_POLICY_{index:04d}"
- OWNER = "wrong-owner"
- FINAL_STATUS = "DRAFT_STATUS_{index:04d}"
- SUMMARY_PREFIX = "legacy-summary-{index:04d}"

Deprecated notes:
The old implementation returned a plain string, stored report data in JSON, and ignored tests.
Do not use this block. It exists only to make the tool output large enough to require compression.
"""


def _report_py() -> str:
    return '''"""Report builder to be implemented by the agent."""

from __future__ import annotations


def build_report() -> dict[str, str]:
    """Return the final audit report metadata."""
    return {}
'''


def _test_report_py() -> str:
    return '''from __future__ import annotations

import unittest

from auditor.report import build_report


class ReportTest(unittest.TestCase):
    def test_build_report_preserves_final_contract(self):
        report = build_report()

        self.assertEqual(report["title"], "Q4-COMPRESS-AUDIT")
        self.assertEqual(report["marker"], "KEEP-COMPRESS-TOOL-9173")
        self.assertEqual(report["policy_code"], "POLICY-COMPRESS-42")
        self.assertEqual(report["owner"], "context-quality")
        self.assertEqual(report["status"], "ready-for-review")
        self.assertTrue(report["summary"].startswith("compress-tool-preserved"))

    def test_build_report_avoids_obsolete_values(self):
        text = repr(build_report())

        self.assertNotIn("OBSOLETE_MARKER", text)
        self.assertNotIn("LEGACY_POLICY", text)
        self.assertNotIn("DRAFT_STATUS", text)
        self.assertNotIn("wrong-owner", text)


if __name__ == "__main__":
    unittest.main()
'''


def _target_readme_md() -> str:
    return """# Compress Tool Pressure Fixture

Implement `auditor.report.build_report()` using the final contract hidden inside
`evidence/noisy_audit_log.txt`.
"""


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create compress_tool pressure eval fixture.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--noise-blocks", type=int, default=DEFAULT_NOISE_BLOCKS)
    args = parser.parse_args(argv)

    root = Path(args.output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = create_case_workspace(root, force=args.force, noise_blocks=args.noise_blocks)
    turns = build_prompt_turns()
    write_prompt_artifacts(root, turns)
    print(json.dumps({"target": str(target), "prompt_stats": prompt_stats(turns, noise_blocks=args.noise_blocks)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
