"""Fixtures and prompts for the native vs LangGraph runner comparison eval."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


CASE_DIRECT_ANSWER = "direct_answer"
CASE_READ_FILE_ANSWER = "read_file_answer"
CASE_GREP_THEN_READ = "grep_then_read"
CASE_SINGLE_FILE_PATCH = "single_file_patch"
CASE_MULTI_FILE_PATCH = "multi_file_patch"
CASE_TOOL_ERROR_RECOVERY = "tool_error_recovery"

CASE_IDS = (
    CASE_DIRECT_ANSWER,
    CASE_READ_FILE_ANSWER,
    CASE_GREP_THEN_READ,
    CASE_SINGLE_FILE_PATCH,
    CASE_MULTI_FILE_PATCH,
    CASE_TOOL_ERROR_RECOVERY,
)


@dataclass(frozen=True)
class PromptTurn:
    kind: str
    text: str


def case_ids(selection: str) -> list[str]:
    if selection == "all":
        return list(CASE_IDS)
    if selection not in CASE_IDS:
        raise ValueError(f"unknown case_id: {selection}")
    return [selection]


def create_case_workspace(root: Path, *, force: bool = False, case_id: str) -> Path:
    target = root / "target_repo"
    if target.exists():
        if not force:
            raise FileExistsError(f"target repo already exists: {target}")
        shutil.rmtree(target)
    target.mkdir(parents=True)

    if case_id == CASE_DIRECT_ANSWER:
        _create_direct_answer_workspace(target)
    elif case_id == CASE_READ_FILE_ANSWER:
        _create_read_file_answer_workspace(target)
    elif case_id == CASE_GREP_THEN_READ:
        _create_grep_then_read_workspace(target)
    elif case_id == CASE_SINGLE_FILE_PATCH:
        _create_single_file_patch_workspace(target)
    elif case_id == CASE_MULTI_FILE_PATCH:
        _create_multi_file_patch_workspace(target)
    elif case_id == CASE_TOOL_ERROR_RECOVERY:
        _create_tool_error_recovery_workspace(target)
    else:
        raise ValueError(f"unknown case_id: {case_id}")

    _write(
        target / ".ai_job_eval_case.json",
        json.dumps({"case_id": case_id}, ensure_ascii=False, indent=2) + "\n",
    )
    _write(target / ".gitignore", "__pycache__/\n*.pyc\n")
    return target


def build_prompt_turns(*, case_id: str) -> list[PromptTurn]:
    if case_id == CASE_DIRECT_ANSWER:
        return [
            PromptTurn(
                kind="direct_answer",
                text=(
                    "直接回答这个校验问题，不要使用工具："
                    "runner 对照基准的验证码是 RUNNER-EVAL-042。"
                    "请只用一句话回答，并包含验证码 RUNNER-EVAL-042。"
                ),
            )
        ]
    if case_id == CASE_READ_FILE_ANSWER:
        return [
            PromptTurn(
                kind="read_file_answer",
                text=(
                    "请读取 workspace 中的 docs/secret_note.txt，然后只用一句话回答里面的 "
                    "release_code 和 owner。"
                ),
            )
        ]
    if case_id == CASE_GREP_THEN_READ:
        return [
            PromptTurn(
                kind="grep_then_read",
                text=(
                    "请在 workspace 中搜索 ROUTING_SENTINEL_GAMMA，找到包含它的文件，"
                    "再读取该文件。最后只用一句话回答 active_route 和 checksum。"
                ),
            )
        ]
    if case_id == CASE_SINGLE_FILE_PATCH:
        return [
            PromptTurn(
                kind="single_file_patch",
                text=(
                    "请修复 calculator.py 中的 apply_discount，使它按 README.md 描述计算。"
                    "你可以读取文件并用 apply_patch 修改。完成后只用一句话说明已修复。"
                ),
            )
        ]
    if case_id == CASE_MULTI_FILE_PATCH:
        return [
            PromptTurn(
                kind="multi_file_patch",
                text=(
                    "请根据 README.md 的要求更新 order_status.py 和 formatter.py，"
                    "让订单状态文案和导出行保持一致。你可以读取文件并用 apply_patch 修改。"
                    "完成后只用一句话说明已完成。"
                ),
            )
        ]
    if case_id == CASE_TOOL_ERROR_RECOVERY:
        return [
            PromptTurn(
                kind="tool_error_recovery",
                text=(
                    "旧说明里说答案在 notes/archive/final_hint.txt。"
                    "请先调用 read_file 读取这个旧路径；如果路径不可用，就在 workspace 中搜索 "
                    "RECOVERY_SENTINEL_88，找到真实文件后读取。最后只用一句话回答 "
                    "recovery_code 和 fallback_owner。"
                ),
            )
        ]
    raise ValueError(f"unknown case_id: {case_id}")


def prompt_texts(turns: Sequence[PromptTurn]) -> list[str]:
    return [turn.text for turn in turns]


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


def prompt_stats(turns: Sequence[PromptTurn], *, case_id: str) -> dict[str, object]:
    prompts = prompt_texts(turns)
    return {
        "case_id": case_id,
        "turn_count": len(turns),
        "prompt_chars": sum(len(text) for text in prompts),
    }


def _create_direct_answer_workspace(target: Path) -> None:
    _write(
        target / "README.md",
        "# Direct Answer Fixture\n\nThis case should not require tool use.\n",
    )


def _create_read_file_answer_workspace(target: Path) -> None:
    _write(
        target / "docs" / "secret_note.txt",
        (
            "Runner comparison note\n"
            "release_code: RF-2026-LG-NATIVE\n"
            "owner: eval-platform\n"
        ),
    )
    _write(target / "README.md", "# Read File Fixture\n\nThe useful fact is in docs/secret_note.txt.\n")


def _create_grep_then_read_workspace(target: Path) -> None:
    _write(target / "notes" / "intro.txt", "No active route is stored here.\n")
    _write(target / "notes" / "archive" / "old_route.txt", "active_route: retired-route\n")
    _write(
        target / "configs" / "routing" / "gamma.toml",
        (
            "# ROUTING_SENTINEL_GAMMA\n"
            "active_route = \"route-gamma-17\"\n"
            "checksum = \"sha256:runner-gamma-4471\"\n"
        ),
    )
    _write(
        target / "README.md",
        "# Grep Fixture\n\nThe active route is stored under configs.\n",
    )


def _create_single_file_patch_workspace(target: Path) -> None:
    _write(
        target / "README.md",
        (
            "# Discount Fixture\n\n"
            "apply_discount(amount, percent) should return amount reduced by percent. "
            "For example, apply_discount(200, 15) must return 170.0.\n"
        ),
    )
    _write(
        target / "calculator.py",
        (
            '"""Small pricing helpers."""\n\n'
            "from __future__ import annotations\n\n\n"
            "def apply_discount(amount: float, percent: float) -> float:\n"
            "    return amount + (amount * percent / 100)\n"
        ),
    )


def _create_multi_file_patch_workspace(target: Path) -> None:
    _write(
        target / "README.md",
        (
            "# Order Status Fixture\n\n"
            "Add a shipped order status. The status code is shipped, the display label is "
            "Shipped to customer, and exported rows should format shipped as "
            "order_id,status_label.\n"
        ),
    )
    _write(
        target / "order_status.py",
        (
            '"""Order status labels."""\n\n'
            "STATUS_LABELS = {\n"
            '    "pending": "Pending review",\n'
            '    "paid": "Paid",\n'
            "}\n\n\n"
            "def label_for_status(status: str) -> str:\n"
            "    return STATUS_LABELS.get(status, \"Unknown\")\n"
        ),
    )
    _write(
        target / "formatter.py",
        (
            '"""Order export formatting."""\n\n'
            "from order_status import label_for_status\n\n\n"
            "def export_order(order: dict[str, str]) -> str:\n"
            "    return f\"{order['id']},{order['status']}\"\n"
        ),
    )


def _create_tool_error_recovery_workspace(target: Path) -> None:
    _write(target / "notes" / "README.md", "Archive moved after the legacy handoff.\n")
    _write(
        target / "knowledge" / "final_hint.txt",
        (
            "RECOVERY_SENTINEL_88\n"
            "recovery_code: RC-88-STABLE\n"
            "fallback_owner: search-first\n"
        ),
    )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a runner comparison eval fixture.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-id", choices=CASE_IDS, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    output = Path(args.output).expanduser().resolve()
    target = create_case_workspace(output, force=args.force, case_id=args.case_id)
    turns = build_prompt_turns(case_id=args.case_id)
    write_prompt_artifacts(output, turns)
    print(json.dumps({"target": str(target), "prompt_stats": prompt_stats(turns, case_id=args.case_id)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
