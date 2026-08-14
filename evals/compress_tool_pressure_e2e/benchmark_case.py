"""Fixtures and prompts for the compress_tool natural pressure eval suite."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


CASE_CONFLICT_CONTRACT_DELAY = "conflict_contract_delay"
CASE_TRACE_DEBUG_DELAY = "trace_debug_delay"
CASE_IDS = (CASE_CONFLICT_CONTRACT_DELAY, CASE_TRACE_DEBUG_DELAY)
CASE_ID = CASE_CONFLICT_CONTRACT_DELAY

DEFAULT_PRESSURE = "hard"
DEFAULT_NOISE_BLOCKS = 1600
PRESSURE_NOISE_BLOCKS = {
    "smoke": 12,
    "medium": 260,
    "hard": DEFAULT_NOISE_BLOCKS,
}


@dataclass(frozen=True)
class PromptTurn:
    kind: str
    text: str


@dataclass(frozen=True)
class PressureConfig:
    name: str
    noise_blocks: int


def pressure_names(selection: str) -> list[str]:
    if selection == "all":
        return list(PRESSURE_NOISE_BLOCKS)
    if selection not in PRESSURE_NOISE_BLOCKS:
        raise ValueError(f"unknown pressure: {selection}")
    return [selection]


def resolve_pressure_config(
    pressure: str = DEFAULT_PRESSURE,
    *,
    noise_blocks: int | None = None,
) -> PressureConfig:
    if pressure not in PRESSURE_NOISE_BLOCKS:
        raise ValueError(f"unknown pressure: {pressure}")
    resolved_noise_blocks = PRESSURE_NOISE_BLOCKS[pressure] if noise_blocks is None else noise_blocks
    if resolved_noise_blocks < 0:
        raise ValueError("noise_blocks must be non-negative")
    return PressureConfig(name=pressure, noise_blocks=resolved_noise_blocks)


def create_case_workspace(
    root: Path,
    *,
    force: bool = False,
    noise_blocks: int | None = None,
    case_id: str = CASE_ID,
    pressure: str = DEFAULT_PRESSURE,
) -> Path:
    config = resolve_pressure_config(pressure, noise_blocks=noise_blocks)
    target = root / "target_repo"
    if target.exists():
        if not force:
            raise FileExistsError(f"target repo already exists: {target}")
        shutil.rmtree(target)

    if case_id == CASE_CONFLICT_CONTRACT_DELAY:
        _create_conflict_contract_workspace(target, config)
    elif case_id == CASE_TRACE_DEBUG_DELAY:
        _create_trace_debug_workspace(target, config)
    else:
        raise ValueError(f"unknown case_id: {case_id}")

    _write(
        target / ".ai_job_eval_case.json",
        json.dumps(
            {
                "case_id": case_id,
                "pressure": config.name,
                "noise_blocks": config.noise_blocks,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    _write(target / ".gitignore", "__pycache__/\n*.pyc\n")
    return target


def build_prompt_turns(
    *,
    case_id: str = CASE_ID,
    pressure: str = DEFAULT_PRESSURE,
) -> list[PromptTurn]:
    config = resolve_pressure_config(pressure)
    if case_id == CASE_CONFLICT_CONTRACT_DELAY:
        return _conflict_contract_prompt_turns(config)
    if case_id == CASE_TRACE_DEBUG_DELAY:
        return _trace_debug_prompt_turns(config)
    raise ValueError(f"unknown case_id: {case_id}")


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


def is_deep_pressure(config: PressureConfig) -> bool:
    return config.name == "hard"


def prompt_stats(
    turns: Sequence[PromptTurn],
    *,
    noise_blocks: int | None = None,
    case_id: str = CASE_ID,
    pressure: str = DEFAULT_PRESSURE,
) -> dict[str, object]:
    config = resolve_pressure_config(pressure, noise_blocks=noise_blocks)
    prompts = prompt_texts(turns)
    return {
        "case_id": case_id,
        "pressure": config.name,
        "depth": "deep" if is_deep_pressure(config) else "standard",
        "turn_count": len(turns),
        "prompt_chars": sum(len(text) for text in prompts),
        "evidence_chars": evidence_chars(case_id=case_id, noise_blocks=config.noise_blocks),
        "noise_blocks": config.noise_blocks,
    }


def evidence_chars(*, case_id: str = CASE_ID, noise_blocks: int = DEFAULT_NOISE_BLOCKS) -> int:
    if case_id == CASE_CONFLICT_CONTRACT_DELAY:
        return (
            len(_conflict_evidence_index())
            + len(_conflict_default_release_packet(noise_blocks=noise_blocks))
            + len(_conflict_errata_index())
            + len(_conflict_contract_decisions(noise_blocks=noise_blocks))
            + len(_conflict_release_overrides(noise_blocks=noise_blocks))
            + len(_conflict_contract_decisions(noise_blocks=noise_blocks))
            + len(_conflict_release_overrides(noise_blocks=noise_blocks))
            + len(_conflict_final_delta())
            + len(_conflict_candidate_delta(noise_blocks=noise_blocks))
            + len(_conflict_shadow_hotfix_matrix(noise_blocks=noise_blocks))
            + len(_conflict_final_delta_draft())
        )
    if case_id == CASE_TRACE_DEBUG_DELAY:
        return (
            len(_trace_evidence_index())
            + len(_trace_default_triage_packet(noise_blocks=noise_blocks))
            + len(_trace_errata_index())
            + len(_trace_production_log(noise_blocks=noise_blocks))
            + len(_trace_state_machine_notes(noise_blocks=noise_blocks))
            + len(_trace_production_log(noise_blocks=noise_blocks))
            + len(_trace_active_manifest())
            + len(_trace_replay_analysis_manifest(noise_blocks=noise_blocks))
            + len(_trace_candidate_state_notes(noise_blocks=noise_blocks))
            + len(_trace_active_manifest_draft())
        )
    raise ValueError(f"unknown case_id: {case_id}")


def noisy_evidence_text(*, noise_blocks: int = DEFAULT_NOISE_BLOCKS) -> str:
    return (
        _conflict_contract_decisions(noise_blocks=noise_blocks)
        + "\n\n"
        + _conflict_release_overrides(noise_blocks=noise_blocks)
    )


def _create_conflict_contract_workspace(target: Path, config: PressureConfig) -> None:
    (target / "auditor").mkdir(parents=True)
    (target / "evidence").mkdir()
    (target / "tests").mkdir()

    _write(target / "auditor" / "__init__.py", '"""Audit fixture package."""\n')
    _write(target / "auditor" / "schema.py", _conflict_schema_py())
    _write(target / "auditor" / "formatting.py", _conflict_formatting_py())
    _write(target / "auditor" / "report.py", _conflict_report_py())
    _write(target / "tests" / "test_report.py", _conflict_test_report_py())
    _write(target / "evidence" / "00_index.txt", _conflict_evidence_index())
    _write(
        target / "evidence" / "01_default_release_packet.txt",
        _conflict_default_release_packet(noise_blocks=config.noise_blocks),
    )
    _write(target / "evidence" / "97_contract_errata_index.txt", _conflict_errata_index())
    _write(
        target / "evidence" / "contract_decisions.txt",
        _conflict_contract_decisions(noise_blocks=config.noise_blocks),
    )
    _write(
        target / "evidence" / "release_overrides.txt",
        _conflict_release_overrides(noise_blocks=config.noise_blocks),
    )
    _write(
        target / "evidence" / "legacy_contract_archive.txt",
        _conflict_contract_decisions(noise_blocks=config.noise_blocks),
    )
    _write(
        target / "evidence" / "draft_release_notes.txt",
        _conflict_release_overrides(noise_blocks=config.noise_blocks),
    )
    _write(target / "evidence" / "final_contract_delta.txt", _conflict_final_delta())
    _write(
        target / "evidence" / "release_candidate_delta.txt",
        _conflict_candidate_delta(noise_blocks=config.noise_blocks),
    )
    _write(
        target / "evidence" / "hotfix_shadow_matrix.txt",
        _conflict_shadow_hotfix_matrix(noise_blocks=config.noise_blocks),
    )
    _write(target / "evidence" / "final_contract_delta_draft.txt", _conflict_final_delta_draft())
    _write(target / "README.md", _conflict_readme_md())


def _create_trace_debug_workspace(target: Path, config: PressureConfig) -> None:
    (target / "reconciler").mkdir(parents=True)
    (target / "evidence").mkdir()
    (target / "tests").mkdir()

    _write(target / "reconciler" / "__init__.py", '"""Trace reconciliation fixture package."""\n')
    _write(target / "reconciler" / "rules.py", _trace_rules_py())
    _write(target / "reconciler" / "decision.py", _trace_decision_py())
    _write(target / "tests" / "test_decision.py", _trace_test_decision_py())
    _write(target / "evidence" / "00_index.txt", _trace_evidence_index())
    _write(
        target / "evidence" / "01_incident_triage_packet.log",
        _trace_default_triage_packet(noise_blocks=config.noise_blocks),
    )
    _write(target / "evidence" / "97_trace_errata_index.txt", _trace_errata_index())
    _write(
        target / "evidence" / "production_trace.log",
        _trace_production_log(noise_blocks=config.noise_blocks),
    )
    _write(
        target / "evidence" / "state_machine_notes.txt",
        _trace_state_machine_notes(noise_blocks=config.noise_blocks),
    )
    _write(
        target / "evidence" / "incident_trace_archive.log",
        _trace_production_log(noise_blocks=config.noise_blocks),
    )
    _write(target / "evidence" / "active_trace_manifest.txt", _trace_active_manifest())
    _write(
        target / "evidence" / "replay_analysis_manifest.txt",
        _trace_replay_analysis_manifest(noise_blocks=config.noise_blocks),
    )
    _write(
        target / "evidence" / "state_machine_candidate_notes.txt",
        _trace_candidate_state_notes(noise_blocks=config.noise_blocks),
    )
    _write(target / "evidence" / "active_trace_manifest_draft.txt", _trace_active_manifest_draft())
    _write(target / "README.md", _trace_readme_md())


def _conflict_contract_prompt_turns(config: PressureConfig) -> list[PromptTurn]:
    turns = [
        PromptTurn(
            kind="default_handoff_research",
            text="""请先做默认交接包调研。

请从 evidence/00_index.txt 开始，只读取索引要求的默认 release 交接包。
本轮不要跟进 post-lock route，不要读取 errata 或其他 evidence 文件。
本轮不要修改文件，不要运行测试；读完后只给简短调研结论，说明默认包是否仍需后续 route 校验，不要粘贴长证据内容。""",
        ),
        PromptTurn(
            kind="errata_research",
            text="""刚才的默认交接包是 pre-lock 资料，现在需要按 post-lock route 纠偏。

请回到默认包里看到的 route，读取对应 errata，并按 errata 的消歧 route 顺序定位有效最终契约依据，不要跳过 route 中要求检查的候选或回滚文件。
本轮不要修改文件，不要运行测试；读完后只给简短纠偏结论，明确哪些默认包事实已作废、后续实现只应保留哪些最终依据，不要粘贴长证据内容。""",
        ),
    ]
    if is_deep_pressure(config):
        turns.append(
            PromptTurn(
                kind="final_contract_consolidation",
                text="""请做一次实现前的最终约束整理。

不要重新扫描整个 evidence 目录；只使用上一轮已经确认的有效最终依据。
请读取 README.md、auditor/report.py、auditor/schema.py、auditor/formatting.py 和 tests/，整理实现必须满足的字段、模块职责和 Python 3.9 兼容约束。
本轮不要修改文件，不要运行测试；只输出简短清单，后续实现必须以这份清单为准。""",
            )
        )
    turns.append(
        PromptTurn(
            kind="implementation",
            text="""现在完成实现。

不要重新读取 evidence 目录；直接使用上一轮证据调研得到的有效最终契约。
请读取 README.md、auditor/report.py、auditor/schema.py、auditor/formatting.py 和 tests/，然后实现 auditor/report.py 里的 build_report()。
保持 report/schema/formatting 的现有职责边界，不要把所有逻辑挤进一个无关的大函数。
不要使用废弃、候选、回滚或错误 owner 的值。
目标运行时是 Python 3.9；不要使用 `str | list[str]` 这类 Python 3.10+ 类型语法或依赖 Python 3.10 的类型别名。
当前 ai_job 的 apply_patch 工具只接受标准 git diff 文本，必须以 diff --git 文件头组织 patch。
不要把 "*** Begin Patch" / "*** End Patch" 格式传给 apply_patch 工具。
完成后简要说明改了哪些文件。""",
        ),
    )
    return turns


def _trace_debug_prompt_turns(config: PressureConfig) -> list[PromptTurn]:
    turns = [
        PromptTurn(
            kind="default_trace_research",
            text="""请先做默认 trace 交接包调研。

请从 evidence/00_index.txt 开始，只读取索引要求的默认 incident triage 包。
本轮不要跟进 post-lock route，不要读取 errata 或其他 evidence 文件。
本轮不要修改文件，不要运行测试；读完后只给简短调研结论，说明默认包是否仍需后续 route 校验，不要粘贴长证据内容。""",
        ),
        PromptTurn(
            kind="trace_errata_research",
            text="""刚才的默认 triage 包是 pre-lock 资料，现在需要按 post-lock route 纠偏。

请回到默认包里看到的 route，读取对应 errata，并按 errata 的消歧 route 顺序定位有效 trace 依据，不要跳过 route 中要求检查的候选或回滚文件；最终依据包括 active 规则和 non-active/default 输出契约。
本轮不要修改文件，不要运行测试；读完后只给简短纠偏结论，明确哪些默认 triage 事实已作废、后续实现只应保留哪些 active 规则和 default 契约，不要粘贴长证据内容。""",
        ),
    ]
    if is_deep_pressure(config):
        turns.append(
            PromptTurn(
                kind="trace_rule_consolidation",
                text="""请做一次修复前的最终 trace 约束整理。

不要重新扫描整个 evidence 目录；只使用上一轮已经确认的 active 规则和 non-active/default 输出契约。
请读取 README.md、reconciler/decision.py、reconciler/rules.py 和 tests/，整理匹配条件、默认分支、模块职责和 Python 3.9 兼容约束。
本轮不要修改文件，不要运行测试；只输出简短清单，后续修复必须以这份清单为准。""",
            )
        )
    turns.append(
        PromptTurn(
            kind="debug_fix",
            text="""现在修复代码。

不要重新读取 evidence 目录；直接使用上一轮 trace 调研得到的有效规则和 non-active/default 输出契约。
请读取 README.md、reconciler/decision.py、reconciler/rules.py 和 tests/，然后修复 build_reconciliation_plan()。
保持 decision/rules 的现有职责边界，不要把所有规则常量挤进一个无关的大函数。
不要使用废弃、候选、回滚或错误 owner 的处理规则。
目标运行时是 Python 3.9；不要使用 `str | list[str]` 这类 Python 3.10+ 类型语法或依赖 Python 3.10 的类型别名。
当前 ai_job 的 apply_patch 工具只接受标准 git diff 文本，必须以 diff --git 文件头组织 patch。
不要把 "*** Begin Patch" / "*** End Patch" 格式传给 apply_patch 工具。
完成后简要说明改了哪些文件。""",
        ),
    )
    return turns


def _conflict_contract_decisions(*, noise_blocks: int) -> str:
    blocks: list[str] = [
        """# Contract Decision Journal

Interpretation rule:
- Prefer records whose channel is release-2026.08.
- Ignore DRAFT, CANDIDATE, OBSOLETE, ROLLED_BACK, and SHADOW records.
- A HOTFIX record overrides an earlier FINAL record only when the same marker is repeated.
""",
        _conflict_core_contract_block("early"),
    ]
    for index in range(1, noise_blocks + 1):
        blocks.append(_conflict_decision_noise_block(index))
        if index == max(1, noise_blocks // 2):
            blocks.append(_conflict_core_contract_block("middle"))
    blocks.append(_conflict_core_contract_block("late"))
    return "\n\n".join(blocks) + "\n"


def _conflict_evidence_index() -> str:
    return """# Evidence Index

Default route:
- Start with 01_default_release_packet.txt.
- Follow any POST_LOCK_ROUTE or ERRATA_ROUTE instruction found inside that packet before implementation.

Do not assume that a file is final because its name contains "final", "active", or "release".
Several nearby files are retained only to reproduce old handoffs and rollback investigations.

Compatibility aliases from older eval revisions:
- contract_decisions.txt mirrors legacy_contract_archive.txt.
- release_overrides.txt mirrors draft_release_notes.txt.
"""


def _conflict_default_release_packet(*, noise_blocks: int) -> str:
    blocks: list[str] = [
        """# Default Release Packet

packet_state = PRE_LOCK_DEFAULT
handoff_owner = release-notes
handoff_confidence = high-before-post-lock-review

This packet intentionally looks complete because it was the default handoff before
the release lock review. It is not safe to implement until the packet footer has
been checked for late routing changes.

[DEFAULT_CONTRACT_BASELINE]
record_state = DEFAULT_ACTIVE_BEFORE_LOCK
channel = release-2026.08
Suggested values before post-lock review:
- REPORT_TITLE = "Q4-COMPRESS-AUDIT-DRAFT"
- REQUIRED_MARKER = "KEEP-CANDIDATE-DEFAULT-1107"
- POLICY_CODE = "POLICY-CANDIDATE-17"
- OWNER = "release-notes"
- FINAL_STATUS = "candidate-ready"
- SUMMARY_PREFIX = "candidate-contract-carried"
- RISK_LEVEL = "medium"
- REVIEW_WINDOW = "2026-W32"
- ESCALATION_CHANNEL = "release-handoff-room"
- PRIMARY_REGION = "iad-3"
- SECONDARY_REGION = "pdx-1"
- CONTROL_PLANE = "cp-candidate-3"
- CONTRACT_VERSION = "ccdl-2026.08-candidate"
- DATASET_FINGERPRINT = "fp-candidate-1107"
- ROUTING_KEYS = ["audit.q4.candidate", "policy.candidate.17", "marker.candidate.1107"]
- EVIDENCE_CHECKSUMS = {"core": "sha256:candidate-core", "override": "sha256:candidate-override", "delta": "sha256:candidate-delta"}
- AUDIT_TAGS = ["candidate-lock", "handoff-review", "prelock"]
- VALIDATION_GATES = ["schema-review", "candidate-owner", "summary-review"]
- OWNER_CHAIN = ["release-notes", "candidate-platform", "handoff-ops"]
- APPROVAL_CHAIN = ["candidate-rfc-1107", "candidate-sec-1107", "candidate-ops-1107"]
- RUNBOOK_STEPS = ["candidate-freeze", "candidate-owner-check", "candidate-publish"]
- WATCHLIST = ["candidate-policy-loader", "candidate-summary-renderer"]
- NOTIFICATION_CHANNELS = ["#release-handoff-room", "#candidate-platform"]
- DEADLINE_MATRIX = {"owner_ack": "2026-08-10T09:00Z", "schema_freeze": "2026-08-11T12:00Z", "release_gate": "2026-08-12T18:00Z"}
- BLOCKING_CONDITIONS = ["pending-post-lock-review"]
- RELEASE_FLAGS = {"requires_manual_review": "no", "allow_legacy_policy": "candidate"}

The default packet was useful for triage, but later errata may supersede it.
""",
    ]
    for index in range(1, noise_blocks + 1):
        blocks.append(_conflict_default_packet_noise_block(index))
        if index in {max(1, noise_blocks // 3), max(1, (noise_blocks * 2) // 3)}:
            blocks.append(_conflict_default_packet_repeated_baseline(index))
    blocks.append(
        """[POST_LOCK_ROUTE_CHANGE]
route_state = SUPERSEDES_THIS_PACKET
route_reason = post-lock contract review replaced the default release packet
ERRATA_ROUTE = evidence/97_contract_errata_index.txt

Everything above this route marker is a withdrawn default handoff unless another
file on the errata route repeats it as active. Values from this packet are a common
source of wrong implementations because they look complete.
"""
    )
    return "\n\n".join(blocks) + "\n"


def _conflict_errata_index() -> str:
    return """# Contract Errata Index

This errata index supersedes evidence/01_default_release_packet.txt.

Disambiguation route, in order:
1. release_candidate_delta.txt captures the withdrawn candidate handoff; inspect it first so candidate values can be explicitly rejected.
2. hotfix_shadow_matrix.txt captures rollback rehearsals; inspect it second so shadow hotfix values can be explicitly rejected.
3. legacy_contract_archive.txt recovers the locked baseline contract.
4. draft_release_notes.txt recovers the active hotfix override rows.
5. final_contract_delta.txt identifies which archive and hotfix facts survive post-lock review.

Withdrawn or investigation-only files:
- release_candidate_delta.txt is a candidate delta from the default handoff.
- hotfix_shadow_matrix.txt contains rollback and shadow hotfix rehearsals.
- final_contract_delta_draft.txt is a pre-lock draft with a misleading name.
- contract_decisions.txt and release_overrides.txt are compatibility aliases, not extra evidence.

Implementation rule:
- Ignore the default release packet after reading this errata.
- Compose only the locked baseline plus active post-lock delta.
"""


def _conflict_final_delta() -> str:
    return """# Final Contract Delta

This short file is authoritative for conflict_contract_delay after the baseline archive has been read.

Use these active source records:
- FINAL_CONTRACT_CORE records from legacy_contract_archive.txt provide title, marker, policy code, risk level, escalation channel, regions, audit tags, and validation gates.
- HOTFIX_CONTRACT_OVERRIDE records from draft_release_notes.txt provide owner, status, summary prefix, review window, owner chain, approval chain, runbook steps, watchlist, notification channels, deadline matrix, blocking conditions, and release flags.

Withdrawn sources:
- contract-candidate records are not active.
- release-override-shadow records are not active.
- Any DRAFT_STATUS, LEGACY_POLICY, wrong-owner, shadow-summary, obsolete blocker, or allow_legacy_policy=yes value is invalid.

Final composition requirement:
- summary must start with "compress-tool-preserved" and mention "KEEP-COMPRESS-TOOL-9173".
- regions must be {"primary": "iad-7", "secondary": "pdx-2"}.
- routing_keys, audit_tags, validation_gates, owner_chain, approval_chain, runbook_steps, watchlist, notification_channels, blocking_conditions, deadline_matrix, and release_flags must preserve the active record ordering.
- handoff_ticket, review_signoffs, rollback_guards, and control_tags are low-salience final fields; they must be preserved exactly.
"""


def _conflict_candidate_delta(*, noise_blocks: int) -> str:
    blocks: list[str] = [
        """# Release Candidate Delta

delta_state = WITHDRAWN_AFTER_POST_LOCK_REVIEW
source_packet = 01_default_release_packet.txt

This file explains the old candidate path. It is retained to make rollback
investigations reproducible, not to drive the final implementation.

[CANDIDATE_DELTA_BASELINE]
- REPORT_TITLE = "Q4-COMPRESS-AUDIT-DRAFT"
- REQUIRED_MARKER = "KEEP-CANDIDATE-DEFAULT-1107"
- POLICY_CODE = "POLICY-CANDIDATE-17"
- OWNER = "release-notes"
- FINAL_STATUS = "candidate-ready"
- SUMMARY_PREFIX = "candidate-contract-carried"
- REVIEW_WINDOW = "2026-W32"
""",
    ]
    for index in range(1, noise_blocks + 1):
        blocks.append(_conflict_candidate_delta_noise_block(index))
    blocks.append(
        """[WITHDRAWAL_FOOTER]
withdrawn_by = final_contract_delta.txt
withdrawal_reason = post-lock review selected KEEP-COMPRESS-TOOL-9173 and context-quality ownership
"""
    )
    return "\n\n".join(blocks) + "\n"


def _conflict_shadow_hotfix_matrix(*, noise_blocks: int) -> str:
    blocks: list[str] = [
        """# Hotfix Shadow Matrix

matrix_state = ROLLBACK_REHEARSAL
This file contains realistic hotfix rows that were used for rehearsal only.
Do not use rows from this matrix for conflict_contract_delay.
""",
    ]
    for index in range(1, noise_blocks + 1):
        blocks.append(_conflict_shadow_hotfix_noise_block(index))
    return "\n\n".join(blocks) + "\n"


def _conflict_final_delta_draft() -> str:
    return """# Final Contract Delta Draft

draft_state = PRE_LOCK_DRAFT_DO_NOT_IMPLEMENT

This filename is intentionally close to final_contract_delta.txt, but this draft
was replaced during post-lock review.

Wrong draft composition:
- summary starts with "candidate-contract-carried"
- owner is "release-notes"
- marker is "KEEP-CANDIDATE-DEFAULT-1107"
- review_window is "2026-W32"
"""


def _conflict_release_overrides(*, noise_blocks: int) -> str:
    blocks: list[str] = [
        """# Release Override Ledger

Only override rows marked ACTIVE_FOR_RELEASE are valid for the final implementation.
Rows marked SHADOW_FINAL are rehearsal data and must not be used.
""",
        _conflict_override_block("early"),
    ]
    for index in range(1, noise_blocks + 1):
        blocks.append(_conflict_override_noise_block(index))
        if index == max(1, noise_blocks // 3):
            blocks.append(_conflict_override_block("middle"))
    blocks.append(_conflict_override_block("late"))
    return "\n\n".join(blocks) + "\n"


def _conflict_core_contract_block(label: str) -> str:
    return f"""[FINAL_CONTRACT_CORE::{label}]
channel = release-2026.08
record_state = ACTIVE_FOR_RELEASE
These values are valid for auditor.report.build_report:
- REPORT_TITLE = "Q4-COMPRESS-AUDIT"
- REQUIRED_MARKER = "KEEP-COMPRESS-TOOL-9173"
- POLICY_CODE = "POLICY-COMPRESS-42"
- RISK_LEVEL = "elevated"
- ESCALATION_CHANNEL = "audit-war-room"
- PRIMARY_REGION = "iad-7"
- SECONDARY_REGION = "pdx-2"
- CONTROL_PLANE = "cp-orion-7"
- CONTRACT_VERSION = "ccdl-2026.08.17"
- DATASET_FINGERPRINT = "fp-9ab4c2-771"
- ROUTING_KEYS = ["audit.q4.retention", "policy.compress.42", "marker.keep.9173"]
- EVIDENCE_CHECKSUMS = {{"core": "sha256:ccdl-core-9173", "override": "sha256:hotfix-42-w33", "delta": "sha256:delta-final-771"}}
- AUDIT_TAGS = ["contract-lock", "manual-review", "q4-retention"]
- VALIDATION_GATES = ["schema-freeze", "owner-ack", "marker-retention", "summary-prefix"]
- CONTROL_TAGS = ["sox-audit", "retention-critical", "post-lock"]

The implementation must expose build_report() -> dict[str, object].
Do not copy OBSOLETE_MARKER, LEGACY_POLICY, DRAFT_STATUS, wrong-owner, or shadow-summary values.
"""


def _conflict_override_block(label: str) -> str:
    return f"""[HOTFIX_CONTRACT_OVERRIDE::{label}]
channel = release-2026.08
record_state = ACTIVE_FOR_RELEASE
marker = KEEP-COMPRESS-TOOL-9173
Valid override values:
- OWNER = "context-quality"
- FINAL_STATUS = "ready-for-review"
- SUMMARY_PREFIX = "compress-tool-preserved"
- REVIEW_WINDOW = "2026-W33"
- OWNER_CHAIN = ["context-quality", "audit-platform", "release-ops"]
- APPROVAL_CHAIN = ["rfc-4172", "sec-1180", "ops-9301"]
- RUNBOOK_STEPS = ["freeze-schema", "notify-context-quality", "verify-marker-retention", "publish-war-room-note"]
- WATCHLIST = ["legacy-policy-loader", "summary-shadow-renderer", "owner-ack-cron"]
- NOTIFICATION_CHANNELS = ["#audit-war-room", "#context-quality"]
- DEADLINE_MATRIX = {{"owner_ack": "2026-08-14T09:00Z", "schema_freeze": "2026-08-15T12:00Z", "release_gate": "2026-08-16T18:00Z"}}
- BLOCKING_CONDITIONS = ["missing-owner-ack", "marker-mismatch", "policy-drift"]
- RELEASE_FLAGS = {{"requires_manual_review": "yes", "allow_legacy_policy": "no"}}
- HANDOFF_TICKET = "HND-9173-W33"
- REVIEW_SIGNOFFS = ["sig-context-9173", "sig-release-42", "sig-audit-771"]
- ROLLBACK_GUARDS = ["deny-legacy-policy", "block-shadow-render", "freeze-marker"]

Summary must start with SUMMARY_PREFIX and mention REQUIRED_MARKER.
"""


def _conflict_decision_noise_block(index: int) -> str:
    return f"""[contract-candidate-{index:04d}]
record_state = SHADOW_FINAL
channel = release-2026.08-draft
This block is intentionally plausible but invalid.
- REPORT_TITLE = "OBSOLETE-AUDIT-{index:04d}"
- REQUIRED_MARKER = "OBSOLETE_MARKER_{index:04d}"
- POLICY_CODE = "LEGACY_POLICY_{index:04d}"
- RISK_LEVEL = "low"
- ESCALATION_CHANNEL = "shadow-channel-{index:04d}"
- PRIMARY_REGION = "legacy-{index:04d}"
- AUDIT_TAGS = ["obsolete", "draft", "shadow"]

Historical note:
The old implementation returned a plain string, stored report data in JSON, and ignored the review window.
This record was rolled back after validation drift and must not affect the final implementation.
"""


def _conflict_default_packet_noise_block(index: int) -> str:
    return f"""[default-handoff-row-{index:04d}]
record_state = DEFAULT_ACTIVE_BEFORE_LOCK
channel = release-2026.08
This row repeats the pre-lock default path and looks valid until the packet footer is read.
- REPORT_TITLE = "Q4-COMPRESS-AUDIT-DRAFT"
- REQUIRED_MARKER = "KEEP-CANDIDATE-DEFAULT-{index:04d}"
- POLICY_CODE = "POLICY-CANDIDATE-{index % 97:02d}"
- OWNER = "release-notes"
- FINAL_STATUS = "candidate-ready"
- SUMMARY_PREFIX = "candidate-contract-carried"
- REVIEW_WINDOW = "2026-W32"
- ESCALATION_CHANNEL = "release-handoff-room"
- PRIMARY_REGION = "iad-3"
- SECONDARY_REGION = "pdx-1"
- RELEASE_FLAGS = {{"requires_manual_review": "no", "allow_legacy_policy": "candidate"}}
- HANDOFF_TICKET = "HND-CANDIDATE-1107"
- REVIEW_SIGNOFFS = ["candidate-signoff"]
- ROLLBACK_GUARDS = ["allow-shadow-summary"]

This row was superseded by the post-lock route marker near the end of the packet.
"""


def _conflict_default_packet_repeated_baseline(index: int) -> str:
    return f"""[default-baseline-repeat-{index:04d}]
record_state = DEFAULT_ACTIVE_BEFORE_LOCK
handoff_note = repeated candidate baseline before post-lock errata
- OWNER_CHAIN = ["release-notes", "candidate-platform", "handoff-ops"]
- BLOCKING_CONDITIONS = ["pending-post-lock-review", "candidate-owner-missing-{index:04d}"]
- AUDIT_TAGS = ["candidate-lock", "handoff-review", "prelock"]
- VALIDATION_GATES = ["schema-review", "candidate-owner", "summary-review"]
"""


def _conflict_candidate_delta_noise_block(index: int) -> str:
    return f"""[candidate-delta-row-{index:04d}]
delta_state = WITHDRAWN_AFTER_POST_LOCK_REVIEW
candidate_marker = KEEP-CANDIDATE-DEFAULT-{index:04d}
candidate_owner = release-notes
candidate_summary = candidate-contract-carried row {index:04d}
candidate_blocker = pending-post-lock-review-{index:04d}
This candidate delta resembles the post-lock delta but is explicitly withdrawn.
"""


def _conflict_shadow_hotfix_noise_block(index: int) -> str:
    return f"""[shadow-hotfix-row-{index:04d}]
record_state = ROLLBACK_REHEARSAL
marker = KEEP-CANDIDATE-DEFAULT-{index:04d}
hotfix_owner = release-notes
status = candidate-ready
summary_prefix = candidate-contract-carried
review_window = 2026-W32
release_flags = {{"requires_manual_review": "no", "allow_legacy_policy": "candidate"}}
handoff_ticket = HND-CANDIDATE-{index:04d}
rollback_guards = ["allow-shadow-summary"]
This rehearsal row is intentionally close to an active hotfix row but was never released.
"""


def _conflict_override_noise_block(index: int) -> str:
    return f"""[release-override-shadow-{index:04d}]
record_state = ROLLED_BACK
marker = OBSOLETE_MARKER_{index:04d}
Candidate override values:
- OWNER = "wrong-owner"
- FINAL_STATUS = "DRAFT_STATUS_{index:04d}"
- SUMMARY_PREFIX = "shadow-summary-{index:04d}"
- REVIEW_WINDOW = "2025-W{index % 52:02d}"
- OWNER_CHAIN = ["wrong-owner", "legacy-ops"]
- BLOCKING_CONDITIONS = ["obsolete-blocker-{index:04d}"]
- RELEASE_FLAGS = {{"requires_manual_review": "no", "allow_legacy_policy": "yes"}}
- HANDOFF_TICKET = "HND-OBSOLETE-{index:04d}"
- REVIEW_SIGNOFFS = ["shadow-signoff"]
- ROLLBACK_GUARDS = ["allow-shadow-summary"]

This row resembles a final override, but it was superseded before release lock.
"""


def _conflict_report_py() -> str:
    return '''"""Report builder to be implemented by the agent."""

from __future__ import annotations

from .formatting import build_summary
from .schema import REPORT_KEYS


def build_report() -> dict[str, object]:
    """Return the final audit report metadata."""
    report: dict[str, object] = {}
    return {key: report[key] for key in REPORT_KEYS}
'''


def _conflict_schema_py() -> str:
    return '''"""Stable report schema for the audit fixture."""

from __future__ import annotations


REPORT_KEYS = [
    "title",
    "marker",
    "policy_code",
    "owner",
    "status",
    "summary",
    "risk_level",
    "review_window",
    "escalation_channel",
    "regions",
    "control_plane",
    "contract_version",
    "dataset_fingerprint",
    "routing_keys",
    "evidence_checksums",
    "audit_tags",
    "validation_gates",
    "owner_chain",
    "approval_chain",
    "runbook_steps",
    "watchlist",
    "notification_channels",
    "deadline_matrix",
    "blocking_conditions",
    "release_flags",
    "handoff_ticket",
    "review_signoffs",
    "rollback_guards",
    "control_tags",
]


STRING_KEYS = [
    "title",
    "marker",
    "policy_code",
    "owner",
    "status",
    "summary",
    "risk_level",
    "review_window",
    "escalation_channel",
    "control_plane",
    "contract_version",
    "dataset_fingerprint",
    "handoff_ticket",
]
'''


def _conflict_formatting_py() -> str:
    return '''"""Formatting helpers for audit reports."""

from __future__ import annotations


def build_summary(prefix: str, marker: str, status: str, owner: str) -> str:
    """Build the stable human-readable report summary."""
    return f"{prefix} {marker} status={status} owner={owner}"
'''


def _conflict_test_report_py() -> str:
    return '''from __future__ import annotations

import unittest

from auditor.report import build_report
from auditor.schema import REPORT_KEYS, STRING_KEYS


class ReportShapeTest(unittest.TestCase):
    def test_build_report_has_required_shape(self):
        report = build_report()

        self.assertEqual(list(report), REPORT_KEYS)
        for key in STRING_KEYS:
            self.assertIsInstance(report[key], str)
            self.assertTrue(report[key])
        self.assertIsInstance(report["regions"], dict)
        self.assertIsInstance(report["routing_keys"], list)
        self.assertIsInstance(report["evidence_checksums"], dict)
        self.assertIsInstance(report["audit_tags"], list)
        self.assertIsInstance(report["validation_gates"], list)
        self.assertIsInstance(report["owner_chain"], list)
        self.assertIsInstance(report["approval_chain"], list)
        self.assertIsInstance(report["runbook_steps"], list)
        self.assertIsInstance(report["watchlist"], list)
        self.assertIsInstance(report["notification_channels"], list)
        self.assertIsInstance(report["deadline_matrix"], dict)
        self.assertIsInstance(report["blocking_conditions"], list)
        self.assertIsInstance(report["release_flags"], dict)
        self.assertIsInstance(report["review_signoffs"], list)
        self.assertIsInstance(report["rollback_guards"], list)
        self.assertIsInstance(report["control_tags"], list)

    def test_build_report_avoids_known_obsolete_prefixes(self):
        text = repr(build_report())

        self.assertNotIn("OBSOLETE_MARKER", text)
        self.assertNotIn("LEGACY_POLICY", text)
        self.assertNotIn("DRAFT_STATUS", text)
        self.assertNotIn("wrong-owner", text)
        self.assertNotIn("shadow-summary", text)
        self.assertNotIn("KEEP-CANDIDATE-DEFAULT", text)
        self.assertNotIn("release-notes", text)
        self.assertNotIn("candidate-contract-carried", text)
        self.assertNotIn("cp-candidate-3", text)
        self.assertNotIn("HND-CANDIDATE", text)
        self.assertNotIn("allow-shadow-summary", text)


if __name__ == "__main__":
    unittest.main()
'''


def _conflict_readme_md() -> str:
    return """# Conflict Contract Delay Fixture

Implement `auditor.report.build_report()` from the valid release contract found
during the evidence research phase.

Keep the generated report ordered according to `auditor.schema.REPORT_KEYS`.
Use `auditor.formatting.build_summary()` for the final summary string.

The target test runtime is Python 3.9.
"""


def _trace_production_log(*, noise_blocks: int) -> str:
    blocks: list[str] = [
        """# Production Trace Archive

Interpretation rule:
- The active incident belongs to tenant aurora-ledger and pipeline delta-sync.
- Ignore replay, dry-run, sandbox, and rollback traces.
- Trace notes are noisy; only CONFIRMED_PRODUCTION_TRACE records are actionable.
""",
        _trace_confirmed_incident_block("early"),
        _trace_secondary_incident_block("early"),
        _trace_audit_incident_block("early"),
    ]
    for index in range(1, noise_blocks + 1):
        blocks.append(_trace_noise_block(index))
        if index == max(1, noise_blocks // 2):
            blocks.append(_trace_confirmed_incident_block("middle"))
        if index == max(1, noise_blocks // 4):
            blocks.append(_trace_secondary_incident_block("middle"))
        if index == max(1, (noise_blocks * 3) // 4):
            blocks.append(_trace_audit_incident_block("middle"))
    blocks.append(_trace_confirmed_incident_block("late"))
    blocks.append(_trace_secondary_incident_block("late"))
    blocks.append(_trace_audit_incident_block("late"))
    return "\n\n".join(blocks) + "\n"


def _trace_evidence_index() -> str:
    return """# Trace Evidence Index

Default route:
- Start with 01_incident_triage_packet.log.
- Follow any POST_LOCK_ROUTE or ERRATA_ROUTE instruction found inside that packet before implementation.

Do not assume that a file is final because its name contains "active", "manifest", or "production".
Several nearby files are retained only to reproduce replay analysis and rollback investigations.

Compatibility alias from older eval revisions:
- production_trace.log mirrors incident_trace_archive.log.
"""


def _trace_default_triage_packet(*, noise_blocks: int) -> str:
    blocks: list[str] = [
        """# Incident Triage Packet

packet_state = PRE_LOCK_TRIAGE_DEFAULT
handoff_owner = triage-desk
handoff_confidence = high-before-post-lock-review

This packet intentionally looks complete because it was the default trace handoff
before release lock review. It is not safe to implement until the packet footer
has been checked for late routing changes.

[DEFAULT_TRIAGE_RULE]
record_state = DEFAULT_ACTIVE_BEFORE_LOCK
tenant = aurora-ledger
pipeline = delta-sync
trace_id = TRACE-CANDIDATE-1107
error_code = E-RETRY-CANDIDATE
attempts = 2
Suggested handling before post-lock review:
- action = "retry-later"
- owner = "triage-desk"
- severity = "sev4"
- retry_after_minutes = "45"
- status = "candidate-retry"
- marker = "TRACE-CANDIDATE-1107"
- queue = "candidate-retry-queue"
- runbook = "rb-candidate-retry"
- escalation_channel = "#triage-desk"
- sla = "PT45M"
- evidence_hash = "sha256:candidate-triage-1107"
- decision_flags = ["candidate", "prelock", "retry"]

The default packet was useful for triage, but later errata may supersede it.
""",
    ]
    for index in range(1, noise_blocks + 1):
        blocks.append(_trace_default_packet_noise_block(index))
        if index in {max(1, noise_blocks // 3), max(1, (noise_blocks * 2) // 3)}:
            blocks.append(_trace_default_packet_repeated_rule(index))
    blocks.append(
        """[POST_LOCK_ROUTE_CHANGE]
route_state = SUPERSEDES_THIS_PACKET
route_reason = post-lock trace review replaced the default triage packet
ERRATA_ROUTE = evidence/97_trace_errata_index.txt

Everything above this route marker is a withdrawn default handoff unless another
file on the errata route repeats it as active. Values from this packet are a common
source of wrong implementations because they look complete.
"""
    )
    return "\n\n".join(blocks) + "\n"


def _trace_errata_index() -> str:
    return """# Trace Errata Index

This errata index supersedes evidence/01_incident_triage_packet.log.

Disambiguation route, in order:
1. replay_analysis_manifest.txt captures withdrawn replay analysis; inspect it first so replay values can be explicitly rejected.
2. state_machine_candidate_notes.txt captures rollback/candidate state rules; inspect it second so candidate state values can be explicitly rejected.
3. incident_trace_archive.log recovers the locked production traces.
4. state_machine_notes.txt recovers the active state rules.
5. active_trace_manifest.txt identifies which trace ids and state rules survive post-lock review.

Withdrawn or investigation-only files:
- replay_analysis_manifest.txt describes replay analysis and candidate handling.
- state_machine_candidate_notes.txt contains rollback and candidate state rules.
- active_trace_manifest_draft.txt is a pre-lock draft with a misleading name.
- production_trace.log is a compatibility alias, not extra evidence.

Implementation rule:
- Ignore the default triage packet after reading this errata.
- Compose only active manifest trace ids plus active state rules.
"""


def _trace_active_manifest() -> str:
    return """# Active Trace Manifest

This short file is authoritative for trace_debug_delay after the archive and state machine notes have been read.

Active production traces for release-2026.08:
- TRACE-KEEP-4821 uses ACTIVE_STATE_RULE.
- TRACE-QUARANTINE-7712 uses ACTIVE_QUARANTINE_RULE.
- TRACE-AUDIT-3345 uses ACTIVE_AUDIT_RULE.

Retired sources:
- trace-replay records are not active.
- candidate-state-rule records are not active.
- retry-later, quarantine-shadow, defer-legacy-audit, legacy-retry, wrong-owner, and TRACE-OBSOLETE values are invalid.

Final composition requirement:
- Preserve the incoming trace_id as marker for each matching active event.
- Active plans must include queue, runbook, escalation_channel, sla, evidence_hash, and decision_flags.
- Non-matching events should keep a non-active default action and preserve the incoming trace_id as marker.
- Active plans must also preserve rule_id, resolver_group, audit_tags, and suppressions for each final rule.
- The default non-active plan must preserve action = "monitor", owner = "triage", severity = "sev4", retry_after_minutes = "15", status = "watching", queue = "triage-watch", runbook = "rb-triage-default", escalation_channel = "#triage", sla = "PT15M", evidence_hash = "sha256:default-monitor", decision_flags = ["default", "non-active"], rule_id = "rule-default-monitor", resolver_group = "triage/oncall", audit_tags = ["default", "non-active"], suppressions = ["active-only"], default_status_code = "NO_ACTIVE_RULE", and default_owner_chain = ["triage", "ledger-watch"].
"""


def _trace_replay_analysis_manifest(*, noise_blocks: int) -> str:
    blocks: list[str] = [
        """# Replay Analysis Manifest

manifest_state = WITHDRAWN_AFTER_POST_LOCK_REVIEW
source_packet = 01_incident_triage_packet.log

This file explains the old replay-analysis path. It is retained to make rollback
investigations reproducible, not to drive the final implementation.

[CANDIDATE_TRACE_RULE]
- trace_id = TRACE-CANDIDATE-1107
- error_code = E-RETRY-CANDIDATE
- action = "retry-later"
- owner = "triage-desk"
- severity = "sev4"
- retry_after_minutes = "45"
- status = "candidate-retry"
""",
    ]
    for index in range(1, noise_blocks + 1):
        blocks.append(_trace_replay_manifest_noise_block(index))
    blocks.append(
        """[WITHDRAWAL_FOOTER]
withdrawn_by = active_trace_manifest.txt
withdrawal_reason = post-lock review selected TRACE-KEEP-4821, TRACE-QUARANTINE-7712, and TRACE-AUDIT-3345
"""
    )
    return "\n\n".join(blocks) + "\n"


def _trace_candidate_state_notes(*, noise_blocks: int) -> str:
    blocks: list[str] = [
        """# State Machine Candidate Notes

notes_state = ROLLBACK_REHEARSAL
This file contains realistic state-machine rows that were used for rehearsal only.
Do not use rows from this file for trace_debug_delay.
""",
    ]
    for index in range(1, noise_blocks + 1):
        blocks.append(_trace_candidate_state_noise_block(index))
    return "\n\n".join(blocks) + "\n"


def _trace_active_manifest_draft() -> str:
    return """# Active Trace Manifest Draft

draft_state = PRE_LOCK_DRAFT_DO_NOT_IMPLEMENT

This filename is intentionally close to active_trace_manifest.txt, but this draft
was replaced during post-lock review.

Wrong draft composition:
- action = "retry-later"
- owner = "triage-desk"
- marker = "TRACE-CANDIDATE-1107"
- retry_after_minutes = "45"
"""


def _trace_state_machine_notes(*, noise_blocks: int) -> str:
    blocks: list[str] = [
        """# State Machine Notes

Only ACTIVE_STATE_RULE entries are valid. Candidate and rollback rules are examples from older incidents.
""",
        _trace_active_rule_block("early"),
        _trace_secondary_rule_block("early"),
        _trace_audit_rule_block("early"),
    ]
    for index in range(1, noise_blocks + 1):
        blocks.append(_trace_rule_noise_block(index))
        if index == max(1, noise_blocks // 3):
            blocks.append(_trace_active_rule_block("middle"))
        if index == max(1, noise_blocks // 5):
            blocks.append(_trace_secondary_rule_block("middle"))
        if index == max(1, (noise_blocks * 4) // 5):
            blocks.append(_trace_audit_rule_block("middle"))
    blocks.append(_trace_active_rule_block("late"))
    blocks.append(_trace_secondary_rule_block("late"))
    blocks.append(_trace_audit_rule_block("late"))
    return "\n\n".join(blocks) + "\n"


def _trace_confirmed_incident_block(label: str) -> str:
    return f"""[CONFIRMED_PRODUCTION_TRACE::{label}]
tenant = aurora-ledger
pipeline = delta-sync
trace_id = TRACE-KEEP-4821
error_code = E-RETRY-9173
attempts = 3
observed_transition = queued -> validated -> retry_timeout -> manual_review_required
release_channel = release-2026.08

The trace_id must be preserved as the marker in the reconciliation plan.
"""


def _trace_secondary_incident_block(label: str) -> str:
    return f"""[CONFIRMED_SECONDARY_TRACE::{label}]
tenant = aurora-ledger
pipeline = delta-sync
trace_id = TRACE-QUARANTINE-7712
error_code = E-CHECKSUM-7712
attempts = 1
observed_transition = queued -> validated -> checksum_mismatch -> quarantine_required
release_channel = release-2026.08

The trace_id must be preserved as the marker in the reconciliation plan.
"""


def _trace_audit_incident_block(label: str) -> str:
    return f"""[CONFIRMED_AUDIT_TRACE::{label}]
tenant = aurora-ledger
pipeline = audit-sync
trace_id = TRACE-AUDIT-3345
error_code = E-AUDIT-LAG-3345
attempts = 2
observed_transition = queued -> audit_lag -> deferred_review_required
release_channel = release-2026.08

The trace_id must be preserved as the marker in the reconciliation plan.
"""


def _trace_active_rule_block(label: str) -> str:
    return f"""[ACTIVE_STATE_RULE::{label}]
Applies when:
- tenant == "aurora-ledger"
- pipeline == "delta-sync"
- error_code == "E-RETRY-9173"
- attempts >= 3

Return plan values:
- action = "open-manual-review"
- owner = "ledger-quality"
- severity = "sev2"
- retry_after_minutes = "0"
- status = "blocked-on-ledger-review"
- marker = trace_id
- queue = "ledger-manual-review"
- runbook = "rb-ledger-9173"
- escalation_channel = "#ledger-quality"
- sla = "PT0M"
- evidence_hash = "sha256:trace-keep-4821"
- decision_flags = ["manual-review", "retain-marker", "block-ledger"]
- rule_id = "rule-ledger-9173"
- resolver_group = "ledger-quality/oncall"
- audit_tags = ["manual-review", "retry-threshold", "release-2026.08"]
- suppressions = ["auto-retry", "candidate-replay"]
"""


def _trace_secondary_rule_block(label: str) -> str:
    return f"""[ACTIVE_QUARANTINE_RULE::{label}]
Applies when:
- tenant == "aurora-ledger"
- pipeline == "delta-sync"
- error_code == "E-CHECKSUM-7712"
- attempts >= 1

Return plan values:
- action = "quarantine-ledger-batch"
- owner = "ledger-integrity"
- severity = "sev1"
- retry_after_minutes = "0"
- status = "blocked-on-integrity-check"
- marker = trace_id
- queue = "ledger-quarantine"
- runbook = "rb-ledger-7712"
- escalation_channel = "#ledger-integrity"
- sla = "PT0M"
- evidence_hash = "sha256:trace-quarantine-7712"
- decision_flags = ["quarantine", "checksum", "block-ledger"]
- rule_id = "rule-ledger-7712"
- resolver_group = "ledger-integrity/oncall"
- audit_tags = ["quarantine", "checksum", "release-2026.08"]
- suppressions = ["partial-replay", "candidate-quarantine"]
"""


def _trace_audit_rule_block(label: str) -> str:
    return f"""[ACTIVE_AUDIT_RULE::{label}]
Applies when:
- tenant == "aurora-ledger"
- pipeline == "audit-sync"
- error_code == "E-AUDIT-LAG-3345"
- attempts >= 2

Return plan values:
- action = "defer-audit-sync"
- owner = "audit-quality"
- severity = "sev3"
- retry_after_minutes = "30"
- status = "deferred-for-audit-window"
- marker = trace_id
- queue = "audit-defer"
- runbook = "rb-audit-3345"
- escalation_channel = "#audit-quality"
- sla = "PT30M"
- evidence_hash = "sha256:trace-audit-3345"
- decision_flags = ["audit-defer", "windowed", "retain-marker"]
- rule_id = "rule-audit-3345"
- resolver_group = "audit-quality/oncall"
- audit_tags = ["audit-defer", "windowed", "release-2026.08"]
- suppressions = ["legacy-audit-retry", "candidate-audit"]
"""


def _trace_noise_block(index: int) -> str:
    return f"""[trace-replay-{index:04d}]
trace_state = replay
tenant = sandbox-ledger-{index:04d}
pipeline = delta-sync
trace_id = TRACE-OBSOLETE-{index:04d}
error_code = E-LEGACY-{index:04d}
attempts = {index % 5}
observed_transition = queued -> validated -> retry_later

Replay traces look similar to the production incident but are not actionable.
Do not preserve TRACE-OBSOLETE markers in the final reconciliation rule.
"""


def _trace_default_packet_noise_block(index: int) -> str:
    return f"""[default-triage-row-{index:04d}]
record_state = DEFAULT_ACTIVE_BEFORE_LOCK
tenant = aurora-ledger
pipeline = delta-sync
trace_id = TRACE-CANDIDATE-{index:04d}
error_code = E-RETRY-CANDIDATE
attempts = {1 + index % 3}
Suggested handling before post-lock review:
- action = "retry-later"
- owner = "triage-desk"
- severity = "sev4"
- retry_after_minutes = "45"
- status = "candidate-retry"
- marker = "TRACE-CANDIDATE-{index:04d}"
- queue = "candidate-retry-queue"
- runbook = "rb-candidate-retry"
- escalation_channel = "#triage-desk"
- sla = "PT45M"
- evidence_hash = "sha256:candidate-triage-{index:04d}"
- decision_flags = ["candidate", "prelock", "retry"]
- rule_id = "rule-candidate-{index:04d}"
- resolver_group = "triage-desk/oncall"
- audit_tags = ["candidate", "prelock"]
- suppressions = ["none"]

This row was superseded by the post-lock route marker near the end of the packet.
"""


def _trace_default_packet_repeated_rule(index: int) -> str:
    return f"""[default-triage-repeat-{index:04d}]
record_state = DEFAULT_ACTIVE_BEFORE_LOCK
handoff_note = repeated candidate triage rule before post-lock errata
- action = "retry-later"
- owner = "triage-desk"
- severity = "sev4"
- retry_after_minutes = "45"
- status = "candidate-retry"
- marker = "TRACE-CANDIDATE-1107"
- queue = "candidate-retry-queue"
- runbook = "rb-candidate-retry"
- escalation_channel = "#triage-desk"
- sla = "PT45M"
- evidence_hash = "sha256:candidate-triage-1107"
- decision_flags = ["candidate", "prelock", "retry"]
- rule_id = "rule-candidate-1107"
- resolver_group = "triage-desk/oncall"
- audit_tags = ["candidate", "prelock"]
- suppressions = ["none"]
"""


def _trace_replay_manifest_noise_block(index: int) -> str:
    return f"""[replay-analysis-row-{index:04d}]
manifest_state = WITHDRAWN_AFTER_POST_LOCK_REVIEW
trace_id = TRACE-CANDIDATE-{index:04d}
error_code = E-RETRY-CANDIDATE
candidate_action = retry-later
candidate_owner = triage-desk
candidate_status = candidate-retry
candidate_queue = candidate-retry-queue
candidate_runbook = rb-candidate-retry
candidate_evidence_hash = sha256:candidate-triage-{index:04d}
This replay-analysis row resembles an active manifest row but was withdrawn.
"""


def _trace_candidate_state_noise_block(index: int) -> str:
    return f"""[candidate-state-rehearsal-{index:04d}]
rule_state = ROLLBACK_REHEARSAL
Candidate handling:
- action = "retry-later"
- owner = "triage-desk"
- severity = "sev4"
- retry_after_minutes = "45"
- status = "candidate-retry"
- marker = "TRACE-CANDIDATE-{index:04d}"
- queue = "candidate-retry-queue"
- runbook = "rb-candidate-retry"
- escalation_channel = "#triage-desk"
- sla = "PT45M"
- evidence_hash = "sha256:candidate-triage-{index:04d}"
- decision_flags = ["candidate", "prelock", "retry"]
- rule_id = "rule-candidate-{index:04d}"
- resolver_group = "triage-desk/oncall"
- audit_tags = ["candidate", "prelock"]
- suppressions = ["none"]

This candidate rule was replaced by the post-lock active state rules.
"""


def _trace_rule_noise_block(index: int) -> str:
    return f"""[candidate-state-rule-{index:04d}]
rule_state = ROLLED_BACK
Candidate handling:
- action = "retry-later"
- alternate_action = "quarantine-shadow"
- audit_action = "defer-legacy-audit"
- owner = "wrong-owner"
- severity = "sev4"
- retry_after_minutes = "{15 + index % 45}"
- status = "legacy-retry"
- marker = "TRACE-OBSOLETE-{index:04d}"
- rule_id = "rule-obsolete-{index:04d}"
- resolver_group = "wrong-owner/oncall"

This rule was retired because it retried ledger failures after the manual-review threshold.
"""


def _trace_decision_py() -> str:
    return '''"""Build reconciliation plans for production trace events."""

from __future__ import annotations

from .rules import DEFAULT_PLAN


def build_reconciliation_plan(event: dict[str, str]) -> dict[str, str]:
    """Return the reconciliation plan for a trace event."""
    plan = dict(DEFAULT_PLAN)
    plan["marker"] = event.get("trace_id", "")
    return plan
'''


def _trace_rules_py() -> str:
    return '''"""Rule constants for production trace reconciliation."""

from __future__ import annotations


PLAN_KEYS = [
    "action",
    "owner",
    "severity",
    "retry_after_minutes",
    "status",
    "marker",
    "queue",
    "runbook",
    "escalation_channel",
    "sla",
    "evidence_hash",
    "decision_flags",
    "rule_id",
    "resolver_group",
    "audit_tags",
    "suppressions",
    "default_status_code",
    "default_owner_chain",
]


DEFAULT_PLAN = {
    "action": "monitor",
    "owner": "triage",
    "severity": "sev4",
    "retry_after_minutes": "15",
    "status": "watching",
    "marker": "",
    "queue": "triage-watch",
    "runbook": "rb-triage-default",
    "escalation_channel": "#triage",
    "sla": "PT15M",
    "evidence_hash": "sha256:default-monitor",
    "decision_flags": ["default", "non-active"],
    "rule_id": "rule-default-monitor",
    "resolver_group": "triage/oncall",
    "audit_tags": ["default", "non-active"],
    "suppressions": ["active-only"],
    "default_status_code": "NO_ACTIVE_RULE",
    "default_owner_chain": ["triage", "ledger-watch"],
}
'''


def _trace_test_decision_py() -> str:
    return '''from __future__ import annotations

import unittest

from reconciler.decision import build_reconciliation_plan
from reconciler.rules import PLAN_KEYS


class ReconciliationPlanShapeTest(unittest.TestCase):
    def test_plan_has_stable_shape(self):
        plan = build_reconciliation_plan(
            {
                "tenant": "sandbox",
                "pipeline": "demo",
                "trace_id": "TRACE-DEMO",
                "error_code": "E-DEMO",
                "attempts": "0",
            }
        )

        self.assertEqual(list(plan), PLAN_KEYS)
        for key, value in plan.items():
            if key in {"decision_flags", "audit_tags", "suppressions", "default_owner_chain"}:
                continue
            self.assertIsInstance(value, str)
        self.assertIsInstance(plan["decision_flags"], list)
        self.assertIsInstance(plan["audit_tags"], list)
        self.assertIsInstance(plan["suppressions"], list)
        self.assertIsInstance(plan["default_owner_chain"], list)

    def test_plan_preserves_input_trace_id_as_marker(self):
        plan = build_reconciliation_plan({"trace_id": "TRACE-DEMO"})

        self.assertEqual(plan["marker"], "TRACE-DEMO")


if __name__ == "__main__":
    unittest.main()
'''


def _trace_readme_md() -> str:
    return """# Trace Debug Delay Fixture

Fix `reconciler.decision.build_reconciliation_plan()` using the active
production trace rule found during the evidence research phase.

Keep the returned plan ordered according to `reconciler.rules.PLAN_KEYS`.
Keep shared rule/default constants in `reconciler.rules`.

The target test runtime is Python 3.9.
"""


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create compress_tool pressure eval fixture.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--case-id", choices=CASE_IDS, default=CASE_ID)
    parser.add_argument("--pressure", choices=tuple(PRESSURE_NOISE_BLOCKS), default=DEFAULT_PRESSURE)
    parser.add_argument("--noise-blocks", type=int, default=None)
    args = parser.parse_args(argv)

    root = Path(args.output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = create_case_workspace(
        root,
        force=args.force,
        noise_blocks=args.noise_blocks,
        case_id=args.case_id,
        pressure=args.pressure,
    )
    turns = build_prompt_turns(case_id=args.case_id, pressure=args.pressure)
    write_prompt_artifacts(root, turns)
    print(
        json.dumps(
            {
                "target": str(target),
                "prompt_stats": prompt_stats(
                    turns,
                    noise_blocks=args.noise_blocks,
                    case_id=args.case_id,
                    pressure=args.pressure,
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
