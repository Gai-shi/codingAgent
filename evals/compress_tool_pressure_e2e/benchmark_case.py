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
DEFAULT_NOISE_BLOCKS = 760
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
    if case_id == CASE_CONFLICT_CONTRACT_DELAY:
        return _conflict_contract_prompt_turns()
    if case_id == CASE_TRACE_DEBUG_DELAY:
        return _trace_debug_prompt_turns()
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
        "turn_count": len(turns),
        "prompt_chars": sum(len(text) for text in prompts),
        "evidence_chars": evidence_chars(case_id=case_id, noise_blocks=config.noise_blocks),
        "noise_blocks": config.noise_blocks,
    }


def evidence_chars(*, case_id: str = CASE_ID, noise_blocks: int = DEFAULT_NOISE_BLOCKS) -> int:
    if case_id == CASE_CONFLICT_CONTRACT_DELAY:
        return (
            len(_conflict_evidence_index())
            + len(_conflict_contract_decisions(noise_blocks=noise_blocks))
            + len(_conflict_release_overrides(noise_blocks=noise_blocks))
            + len(_conflict_final_delta())
        )
    if case_id == CASE_TRACE_DEBUG_DELAY:
        return (
            len(_trace_evidence_index())
            + len(_trace_production_log(noise_blocks=noise_blocks))
            + len(_trace_state_machine_notes(noise_blocks=noise_blocks))
            + len(_trace_active_manifest())
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
    _write(target / "auditor" / "report.py", _conflict_report_py())
    _write(target / "tests" / "test_report.py", _conflict_test_report_py())
    _write(target / "evidence" / "00_index.txt", _conflict_evidence_index())
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
    _write(target / "README.md", _conflict_readme_md())


def _create_trace_debug_workspace(target: Path, config: PressureConfig) -> None:
    (target / "reconciler").mkdir(parents=True)
    (target / "evidence").mkdir()
    (target / "tests").mkdir()

    _write(target / "reconciler" / "__init__.py", '"""Trace reconciliation fixture package."""\n')
    _write(target / "reconciler" / "decision.py", _trace_decision_py())
    _write(target / "tests" / "test_decision.py", _trace_test_decision_py())
    _write(target / "evidence" / "00_index.txt", _trace_evidence_index())
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
    _write(target / "README.md", _trace_readme_md())


def _conflict_contract_prompt_turns() -> list[PromptTurn]:
    return [
        PromptTurn(
            kind="evidence_research",
            text="""请先做证据调研。

请从 evidence/00_index.txt 开始，根据索引自行定位实现依据。
材料里包含很长的历史档案、草稿说明和较短的最终 delta；不要凭文件名直接猜最终来源。
本轮不要修改文件，不要运行测试；读完只回复“调研完成”，不要粘贴证据内容。""",
        ),
        PromptTurn(
            kind="implementation",
            text="""现在完成实现。

不要重新读取 evidence 目录；直接使用上一轮证据调研得到的有效最终契约。
请读取 README.md、auditor/report.py 和 tests/，然后实现 auditor/report.py 里的 build_report()。
不要使用废弃、候选、回滚或错误 owner 的值。
当前 ai_job 的 apply_patch 工具只接受标准 git diff 文本，必须以 diff --git 文件头组织 patch。
不要把 "*** Begin Patch" / "*** End Patch" 格式传给 apply_patch 工具。
完成后简要说明改了哪些文件。""",
        ),
    ]


def _trace_debug_prompt_turns() -> list[PromptTurn]:
    return [
        PromptTurn(
            kind="trace_research",
            text="""请先做生产 trace 调研。

请从 evidence/00_index.txt 开始，根据索引自行定位 active trace 依据。
材料里包含很长的 incident archive、过期状态机规则和较短的 active manifest；不要凭文件名直接猜最终来源。
本轮不要修改文件，不要运行测试；读完只回复“调研完成”，不要粘贴证据内容。""",
        ),
        PromptTurn(
            kind="debug_fix",
            text="""现在修复代码。

不要重新读取 evidence 目录；直接使用上一轮 trace 调研得到的有效规则。
请读取 README.md、reconciler/decision.py 和 tests/，然后修复 build_reconciliation_plan()。
不要使用废弃、候选、回滚或错误 owner 的处理规则。
当前 ai_job 的 apply_patch 工具只接受标准 git diff 文本，必须以 diff --git 文件头组织 patch。
不要把 "*** Begin Patch" / "*** End Patch" 格式传给 apply_patch 工具。
完成后简要说明改了哪些文件。""",
        ),
    ]


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

Start with the large archive when recovering the baseline contract:
- legacy_contract_archive.txt: long baseline archive with final core fields mixed with obsolete candidates.
- draft_release_notes.txt: long release note ledger; many rows were withdrawn or rolled back.
- final_contract_delta.txt: short delta that identifies which archive facts remain active and which release-note overrides win.

Do not implement directly from the archive alone. The final implementation is baseline archive facts plus active delta overrides.

Compatibility aliases from older eval revisions:
- contract_decisions.txt mirrors legacy_contract_archive.txt.
- release_overrides.txt mirrors draft_release_notes.txt.
"""


def _conflict_final_delta() -> str:
    return """# Final Contract Delta

This short file is authoritative for conflict_contract_delay after the baseline archive has been read.

Use these active source records:
- FINAL_CONTRACT_CORE records from legacy_contract_archive.txt provide title, marker, policy code, risk level, escalation channel, regions, audit tags, and validation gates.
- HOTFIX_CONTRACT_OVERRIDE records from draft_release_notes.txt provide owner, status, summary prefix, review window, owner chain, blocking conditions, and release flags.

Withdrawn sources:
- contract-candidate records are not active.
- release-override-shadow records are not active.
- Any DRAFT_STATUS, LEGACY_POLICY, wrong-owner, shadow-summary, obsolete blocker, or allow_legacy_policy=yes value is invalid.

Final composition requirement:
- summary must start with "compress-tool-preserved" and mention "KEEP-COMPRESS-TOOL-9173".
- regions must be {"primary": "iad-7", "secondary": "pdx-2"}.
- audit_tags, validation_gates, owner_chain, blocking_conditions, and release_flags must preserve the active record ordering.
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
- AUDIT_TAGS = ["contract-lock", "manual-review", "q4-retention"]
- VALIDATION_GATES = ["schema-freeze", "owner-ack", "marker-retention", "summary-prefix"]

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
- BLOCKING_CONDITIONS = ["missing-owner-ack", "marker-mismatch", "policy-drift"]
- RELEASE_FLAGS = {{"requires_manual_review": "yes", "allow_legacy_policy": "no"}}

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

This row resembles a final override, but it was superseded before release lock.
"""


def _conflict_report_py() -> str:
    return '''"""Report builder to be implemented by the agent."""

from __future__ import annotations


def build_report() -> dict[str, object]:
    """Return the final audit report metadata."""
    return {}
'''


def _conflict_test_report_py() -> str:
    return '''from __future__ import annotations

import unittest

from auditor.report import build_report


class ReportShapeTest(unittest.TestCase):
    def test_build_report_has_required_shape(self):
        report = build_report()

        required_keys = {
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
            "audit_tags",
            "validation_gates",
            "owner_chain",
            "blocking_conditions",
            "release_flags",
        }
        self.assertEqual(set(report), required_keys)
        for key in {
            "title",
            "marker",
            "policy_code",
            "owner",
            "status",
            "summary",
            "risk_level",
            "review_window",
            "escalation_channel",
        }:
            self.assertIsInstance(report[key], str)
            self.assertTrue(report[key])
        self.assertIsInstance(report["regions"], dict)
        self.assertIsInstance(report["audit_tags"], list)
        self.assertIsInstance(report["validation_gates"], list)
        self.assertIsInstance(report["owner_chain"], list)
        self.assertIsInstance(report["blocking_conditions"], list)
        self.assertIsInstance(report["release_flags"], dict)

    def test_build_report_avoids_known_obsolete_prefixes(self):
        text = repr(build_report())

        self.assertNotIn("OBSOLETE_MARKER", text)
        self.assertNotIn("LEGACY_POLICY", text)
        self.assertNotIn("DRAFT_STATUS", text)
        self.assertNotIn("wrong-owner", text)
        self.assertNotIn("shadow-summary", text)


if __name__ == "__main__":
    unittest.main()
'''


def _conflict_readme_md() -> str:
    return """# Conflict Contract Delay Fixture

Implement `auditor.report.build_report()` from the valid release contract found
during the evidence research phase.
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

Start with the archive when recovering production trace context:
- incident_trace_archive.log: long incident archive with confirmed traces mixed with replay and sandbox traces.
- state_machine_notes.txt: long rule ledger with active rules mixed with rollback candidates.
- active_trace_manifest.txt: short manifest that identifies which trace ids and state rules are active for this release.

Do not implement directly from the archive alone. The final implementation is active manifest ids plus active state rules.

Compatibility alias from older eval revisions:
- production_trace.log mirrors incident_trace_archive.log.
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
- Non-matching events should keep a non-active default action and preserve the incoming trace_id as marker.
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

This rule was retired because it retried ledger failures after the manual-review threshold.
"""


def _trace_decision_py() -> str:
    return '''"""Build reconciliation plans for production trace events."""

from __future__ import annotations


def build_reconciliation_plan(event: dict[str, str]) -> dict[str, str]:
    """Return the reconciliation plan for a trace event."""
    return {
        "action": "monitor",
        "owner": "triage",
        "severity": "sev4",
        "retry_after_minutes": "15",
        "status": "watching",
        "marker": event.get("trace_id", ""),
    }
'''


def _trace_test_decision_py() -> str:
    return '''from __future__ import annotations

import unittest

from reconciler.decision import build_reconciliation_plan


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

        self.assertEqual(
            set(plan),
            {
                "action",
                "owner",
                "severity",
                "retry_after_minutes",
                "status",
                "marker",
            },
        )
        for value in plan.values():
            self.assertIsInstance(value, str)

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
