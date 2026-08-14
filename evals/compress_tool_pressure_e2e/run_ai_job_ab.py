"""Run a real ai_job A/B pressure eval suite for compress_tool."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence, TextIO

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark_case import (
        CASE_IDS,
        DEFAULT_PRESSURE,
        PRESSURE_NOISE_BLOCKS,
        build_prompt_turns,
        create_case_workspace,
        pressure_names,
        prompt_stats,
        prompt_texts,
        write_prompt_artifacts,
    )
    from grader import grade_target
else:
    from .benchmark_case import (
        CASE_IDS,
        DEFAULT_PRESSURE,
        PRESSURE_NOISE_BLOCKS,
        build_prompt_turns,
        create_case_workspace,
        pressure_names,
        prompt_stats,
        prompt_texts,
        write_prompt_artifacts,
    )
    from .grader import grade_target


TOOL_POLICY_NEUTRAL = "neutral"
TOOL_POLICY_GUIDED = "guided"
TOOL_POLICIES = (TOOL_POLICY_NEUTRAL, TOOL_POLICY_GUIDED)
AI_JOB_TRACE_LOG_PATH_ENV = "AI_JOB_TRACE_LOG_PATH"
AI_JOB_SESSION_RECORD_PATH_ENV = "AI_JOB_SESSION_RECORD_PATH"

GUIDED_COMPRESS_TOOL_HINT = """工具使用提示：
当前环境提供 compress_tool。读取长 evidence 或发现之前读取的是 pre-lock/误导资料后，如果你已经抽取出后续需要保留的事实，可以自主调用 compress_tool，把已读的冗长工具输出替换成短摘要，再继续完成任务；是否调用由你判断。
如果调用 compress_tool，摘要要保留后续实现可能需要的 exact key/value、触发条件、默认/兜底分支、低显著字段、字段顺序/结构要求，以及哪些候选或旧事实已作废。不要只写结论性摘要。"""

_RUN_LOG_LOCK = threading.Lock()
_PROGRESS_LOCK = threading.Lock()


@dataclass(frozen=True)
class SessionSection:
    title: str
    language: str
    content: str


@dataclass(frozen=True)
class VariantTask:
    output: Path
    turns: Sequence[object]
    variant: str
    disable_compress_tool: bool
    case_id: str
    pressure: str
    tool_policy: str


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run compress_tool pressure A/B eval against ai_job.")
    parser.add_argument("--output", required=True, help="Benchmark output directory.")
    parser.add_argument("--force", action="store_true", help="Replace output directory if it exists.")
    parser.add_argument("--case-id", choices=tuple(["all", *CASE_IDS]), default="all")
    parser.add_argument("--pressure", choices=tuple(["all", *PRESSURE_NOISE_BLOCKS]), default=DEFAULT_PRESSURE)
    parser.add_argument(
        "--tool-policy",
        choices=tuple(["all", *TOOL_POLICIES]),
        default=TOOL_POLICY_NEUTRAL,
        help=(
            "Prompt policy for compress_tool usage. neutral keeps enabled/disabled prompts identical; "
            "guided adds a light compress_tool hint only to enabled runs."
        ),
    )
    parser.add_argument(
        "--noise-blocks",
        type=int,
        default=None,
        help=(
            "Compatibility override for generated evidence size. When omitted, each pressure "
            "uses its configured default."
        ),
    )
    parser.add_argument(
        "--ai-job-command",
        default=f"{sys.executable} -m ai_job",
        help="Command used to start ai_job, without --workspace.",
    )
    parser.add_argument(
        "--ai-job-source-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Path added to PYTHONPATH when launching ai_job.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of ai_job variant processes to run concurrently.",
    )
    parser.add_argument(
        "--auto-compression-context-window",
        type=int,
        default=10_000_000,
        help=(
            "Temporary AI_JOB_CONTEXT_WINDOW used during this eval so automatic "
            "context compression does not mask compress_tool behavior."
        ),
    )
    parser.add_argument("--progress-interval-seconds", type=float, default=5.0)
    parser.add_argument(
        "--progress",
        dest="no_progress",
        action="store_false",
        help="Show live per-variant progress in the terminal.",
    )
    parser.add_argument(
        "--no-progress",
        dest="no_progress",
        action="store_true",
        default=True,
        help="Keep terminal output compact and write progress to run_ai_job_ab.log.",
    )
    args = parser.parse_args(argv)

    output = Path(args.output).expanduser().resolve()
    if output.exists() and args.force:
        import shutil

        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    run_log_path = output / "run_ai_job_ab.log"

    case_ids = list(CASE_IDS) if args.case_id == "all" else [args.case_id]
    selected_pressures = pressure_names(args.pressure)
    selected_tool_policies = _tool_policy_names(args.tool_policy)
    if args.max_workers < 1:
        parser.error("--max-workers must be >= 1")

    multi_policy = len(selected_tool_policies) > 1
    policy_results: dict[str, dict[str, object]] = {}
    with run_log_path.open("w", encoding="utf-8") as run_log:
        _write_run_log(run_log, "ai_job compress_tool pressure eval started")
        _write_run_log(run_log, f"output={output}")
        _write_run_log(run_log, f"case_ids={case_ids}")
        _write_run_log(run_log, f"pressures={selected_pressures}")
        _write_run_log(run_log, f"tool_policies={selected_tool_policies}")
        _write_run_log(run_log, f"max_workers={args.max_workers}")
        for tool_policy in selected_tool_policies:
            policy_output = output / tool_policy if multi_policy else output
            policy_output.mkdir(parents=True, exist_ok=True)
            cases = _run_policy_suite(
                args,
                output=policy_output,
                case_ids=case_ids,
                selected_pressures=selected_pressures,
                tool_policy=tool_policy,
                run_log=run_log,
            )
            policy_results[tool_policy] = {
                "tool_policy": tool_policy,
                "cases": cases,
                "summary": _summarize_suite(cases),
            }

    single_policy_result = policy_results[selected_tool_policies[0]] if len(selected_tool_policies) == 1 else None

    result = {
        "runner": "ai_job_compress_tool_pressure_suite",
        "case_ids": case_ids,
        "pressures": selected_pressures,
        "tool_policy": selected_tool_policies[0] if len(selected_tool_policies) == 1 else "all",
        "tool_policies": selected_tool_policies,
        "noise_blocks_override": args.noise_blocks,
        "run_log": str(run_log_path),
        "policy_results": policy_results,
        "cases": single_policy_result["cases"] if single_policy_result is not None else None,
        "summary": (
            single_policy_result["summary"]
            if single_policy_result is not None
            else _summarize_policy_results(policy_results)
        ),
    }
    result_path = output / "result_compress_tool_ab.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _append_result_summary_log(run_log_path, result_path, result)
    print(_format_run_summary(result_path, run_log_path, result))
    return 0 if result["summary"]["compress_tool_helped_count"] > 0 else 1


def _tool_policy_names(selection: str) -> list[str]:
    if selection == "all":
        return list(TOOL_POLICIES)
    if selection not in TOOL_POLICIES:
        raise ValueError(f"unknown tool_policy: {selection}")
    return [selection]


def _write_run_log(stream: TextIO, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with _RUN_LOG_LOCK:
        stream.write(f"[{timestamp}] {message}\n")
        stream.flush()


def _append_result_summary_log(path: Path, result_path: Path, result: dict[str, object]) -> None:
    summary = result["summary"]
    with path.open("a", encoding="utf-8") as stream:
        _write_run_log(stream, f"result={result_path}")
        _write_run_log(
            stream,
            (
                "summary "
                f"enabled_success_rate={summary['enabled_success_rate']} "
                f"disabled_success_rate={summary['disabled_success_rate']} "
                f"correctness_delta={summary['correctness_delta']} "
                f"compress_tool_helped_count={summary['compress_tool_helped_count']} "
                f"enabled_compress_tool_call_count={summary['enabled_compress_tool_call_count']}"
            ),
        )


def _format_run_summary(result_path: Path, run_log_path: Path, result: dict[str, object]) -> str:
    summary = result["summary"]
    lines = [
        "compress_tool pressure eval completed",
        f"result_json: {result_path}",
        f"run_log: {run_log_path}",
        (
            "summary: "
            f"enabled_success_rate={summary['enabled_success_rate']}, "
            f"disabled_success_rate={summary['disabled_success_rate']}, "
            f"correctness_delta={summary['correctness_delta']}, "
            f"compress_tool_helped_count={summary['compress_tool_helped_count']}, "
            f"enabled_compress_tool_call_count={summary['enabled_compress_tool_call_count']}"
        ),
    ]
    pure_delta = summary.get("pure_availability_delta")
    guided_delta = summary.get("guided_tool_delta")
    if pure_delta is not None or guided_delta is not None:
        lines.append(f"policy_deltas: pure_availability_delta={pure_delta}, guided_tool_delta={guided_delta}")
    return "\n".join(lines)


def _run_policy_suite(
    args: argparse.Namespace,
    *,
    output: Path,
    case_ids: Sequence[str],
    selected_pressures: Sequence[str],
    tool_policy: str,
    run_log: TextIO,
) -> dict[str, dict[str, object]]:
    cell_inputs: dict[tuple[str, str], dict[str, object]] = {}
    variant_tasks: list[VariantTask] = []
    for case_id in case_ids:
        for pressure in selected_pressures:
            case_pressure_output = output / case_id / pressure
            case_pressure_output.mkdir(parents=True, exist_ok=True)
            turns = build_prompt_turns(case_id=case_id, pressure=pressure)
            write_prompt_artifacts(case_pressure_output, turns)
            cell_inputs[(case_id, pressure)] = {
                "case_id": case_id,
                "pressure": pressure,
                "tool_policy": tool_policy,
                "turns": turns,
            }
            variant_tasks.append(
                VariantTask(
                    output=case_pressure_output,
                    turns=turns,
                    variant="enabled",
                    disable_compress_tool=False,
                    case_id=case_id,
                    pressure=pressure,
                    tool_policy=tool_policy,
                )
            )
            variant_tasks.append(
                VariantTask(
                    output=case_pressure_output,
                    turns=turns,
                    variant="disabled",
                    disable_compress_tool=True,
                    case_id=case_id,
                    pressure=pressure,
                    tool_policy=tool_policy,
                )
            )

    _write_run_log(run_log, f"{tool_policy}: scheduling {len(variant_tasks)} variant runs")
    variant_results = _run_variant_tasks(args, variant_tasks, run_log)

    cases: dict[str, dict[str, object]] = {}
    for case_id in case_ids:
        case_results: dict[str, object] = {}
        for pressure in selected_pressures:
            cell = cell_inputs[(case_id, pressure)]
            turns = cell["turns"]
            enabled = variant_results[(tool_policy, case_id, pressure, "enabled")]
            disabled = variant_results[(tool_policy, case_id, pressure, "disabled")]
            comparison = _compare(enabled, disabled)
            case_results[pressure] = {
                "case_id": case_id,
                "pressure": pressure,
                "tool_policy": tool_policy,
                "prompt_stats": {
                    **prompt_stats(
                        turns,
                        noise_blocks=args.noise_blocks,
                        case_id=case_id,
                        pressure=pressure,
                    ),
                    "enabled_prompt_chars": enabled["prompt_chars"],
                    "disabled_prompt_chars": disabled["prompt_chars"],
                },
                "enabled": enabled,
                "disabled": disabled,
                "comparison": comparison,
            }
        cases[case_id] = case_results
    return cases


def _run_variant_tasks(
    args: argparse.Namespace,
    tasks: Sequence[VariantTask],
    run_log: TextIO,
) -> dict[tuple[str, str, str, str], dict[str, object]]:
    results: dict[tuple[str, str, str, str], dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(_run_variant, args, task, run_log): task
            for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            result = future.result()
            key = (task.tool_policy, task.case_id, task.pressure, task.variant)
            results[key] = result
    return results


def _run_variant(
    args: argparse.Namespace,
    task: VariantTask,
    run_log: TextIO,
) -> dict[str, object]:
    output = task.output
    turns = task.turns
    variant = task.variant
    disable_compress_tool = task.disable_compress_tool
    case_id = task.case_id
    pressure = task.pressure
    tool_policy = task.tool_policy
    variant_dir = output / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    target = create_case_workspace(
        variant_dir,
        force=True,
        noise_blocks=args.noise_blocks,
        case_id=case_id,
        pressure=pressure,
    )

    effective_prompts = _effective_prompt_texts(
        turns,
        tool_policy=tool_policy,
        disable_compress_tool=disable_compress_tool,
    )
    (variant_dir / "stdin_prompts.txt").write_text(
        "\n\n--- prompt turn ---\n\n".join(effective_prompts) + "\n",
        encoding="utf-8",
    )
    prompts = [_flatten_for_line_cli(text) for text in effective_prompts]
    stdin_text = "\n".join(prompts + ["exit"]) + "\n"
    cmd = shlex.split(args.ai_job_command) + ["--workspace", str(target)]
    if disable_compress_tool:
        cmd.append("--disable-compress-tool")

    env = os.environ.copy()
    source_root = str(Path(args.ai_job_source_root).expanduser().resolve())
    trace_log_base_path = variant_dir / "ai_job_run" / "logs" / "log.log"
    session_record_base_path = variant_dir / "ai_job_run" / "sessions" / "sessions.md"
    env["PYTHONPATH"] = source_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["AI_JOB_CONTEXT_WINDOW"] = str(args.auto_compression_context_window)
    env[AI_JOB_TRACE_LOG_PATH_ENV] = str(trace_log_base_path)
    env[AI_JOB_SESSION_RECORD_PATH_ENV] = str(session_record_base_path)

    started_at = time.monotonic()
    label = f"{tool_policy}/{case_id}/{pressure}/{variant}"
    _write_run_log(run_log, f"{label} started")
    completed = _run_command_with_progress(
        cmd,
        cwd=source_root,
        env=env,
        stdin_text=stdin_text,
        timeout_seconds=args.timeout_seconds,
        progress_interval_seconds=args.progress_interval_seconds,
        show_progress=not args.no_progress,
        label=label,
        log_stream=run_log,
    )
    elapsed_seconds = round(time.monotonic() - started_at, 3)

    (variant_dir / "ai_job_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (variant_dir / "ai_job_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    diagnostics = collect_run_diagnostics(completed.stdout, completed.stderr)
    diagnostics["elapsed_seconds"] = elapsed_seconds
    grade = grade_target(target, case_id=case_id)
    variant_result = {
        "case_id": case_id,
        "pressure": pressure,
        "tool_policy": tool_policy,
        "variant": variant,
        "disable_compress_tool": disable_compress_tool,
        "guidance_injected": _should_inject_guided_hint(
            tool_policy=tool_policy,
            disable_compress_tool=disable_compress_tool,
        ),
        "prompt_chars": sum(len(text) for text in effective_prompts),
        "command": cmd,
        "exit_code": completed.returncode,
        "elapsed_seconds": elapsed_seconds,
        "target": str(target),
        "trace_log_base_path": str(trace_log_base_path),
        "session_record_base_path": str(session_record_base_path),
        "diagnostics": diagnostics,
        "grade": asdict(grade),
    }
    (variant_dir / f"result_{variant}.json").write_text(
        json.dumps(variant_result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_run_log(
        run_log,
        (
            f"{label} graded passed={grade.passed} score={grade.score} "
            f"compress_calls={diagnostics['compress_tool_call_count']} "
            f"saved_chars={diagnostics['estimated_tool_context_chars_saved']}"
        ),
    )
    return variant_result


def _effective_prompt_texts(
    turns: Sequence[object],
    *,
    tool_policy: str,
    disable_compress_tool: bool,
) -> list[str]:
    texts = prompt_texts(turns)
    if not _should_inject_guided_hint(tool_policy=tool_policy, disable_compress_tool=disable_compress_tool):
        return texts
    return [f"{GUIDED_COMPRESS_TOOL_HINT}\n\n{text}" for text in texts]


def _should_inject_guided_hint(*, tool_policy: str, disable_compress_tool: bool) -> bool:
    return tool_policy == TOOL_POLICY_GUIDED and not disable_compress_tool


def collect_run_diagnostics(stdout: str, stderr: str) -> dict[str, object]:
    session_record = _extract_banner_path(stdout, "session_record")
    log_file = _extract_banner_path(stdout, "log_file")
    session_text = _read_text_if_present(session_record)
    sections = list(_iter_session_sections(session_text))
    read_file_chars = _tool_result_text_chars(sections, "read_file")
    read_file_chars_by_signature = _tool_result_chars_by_call_signature(
        sections,
        tool_name="read_file",
    )
    compress_stats = _compress_replacement_stats(sections)
    return {
        "log_file": log_file,
        "session_record": session_record,
        "session_record_chars": len(session_text),
        "llm_request_failed_count": stderr.count("LLM 请求失败"),
        "context_length_exceeded_count": stderr.count("context_length_exceeded"),
        "python_traceback_count": stderr.count("Traceback (most recent call last):"),
        "user_message_count": len(re.findall(r"^## .* UserMessage", session_text, re.MULTILINE)),
        "tool_call_count": len(re.findall(r"^## .* ToolCall ", session_text, re.MULTILINE)),
        "read_file_tool_call_count": len(re.findall(r"^## .* ToolCall read_file", session_text, re.MULTILINE)),
        "apply_patch_tool_call_count": len(re.findall(r"^## .* ToolCall apply_patch", session_text, re.MULTILINE)),
        "compress_tool_call_count": len(re.findall(r"^## .* ToolCall compress_tool", session_text, re.MULTILINE)),
        "compress_tool_success_count": compress_stats["success_count"],
        "compress_tool_error_count": compress_stats["error_count"],
        "compress_tool_errors": compress_stats["errors"],
        "tool_error_count": len(re.findall(r"\nError: ", session_text)),
        "read_file_tool_result_chars": read_file_chars,
        "largest_read_file_tool_result_chars": max(
            (max(chars) for chars in read_file_chars_by_signature.values()),
            default=0,
        ),
        "compress_replacement_chars": compress_stats["attempted_replacement_chars"],
        "successful_compress_replacement_chars": compress_stats["successful_replacement_chars"],
        "estimated_tool_context_chars_saved": compress_stats["estimated_tool_context_chars_saved"],
    }


def _compare(enabled: dict[str, object], disabled: dict[str, object]) -> dict[str, object]:
    enabled_grade = enabled["grade"]
    disabled_grade = disabled["grade"]
    enabled_passed = bool(enabled_grade["passed"])
    disabled_passed = bool(disabled_grade["passed"])
    enabled_diagnostics = enabled["diagnostics"]
    disabled_diagnostics = disabled["diagnostics"]
    enabled_compress_tool_success_count = int(enabled_diagnostics.get("compress_tool_success_count", 0))
    enabled_saved_chars = int(enabled_diagnostics.get("estimated_tool_context_chars_saved", 0))
    enabled_compress_tool_effective = enabled_compress_tool_success_count > 0 and enabled_saved_chars > 0
    return {
        "verdict": _comparison_verdict(
            enabled_passed=enabled_passed,
            disabled_passed=disabled_passed,
            enabled_compress_tool_call_count=int(enabled_diagnostics.get("compress_tool_call_count", 0)),
            enabled_compress_tool_success_count=enabled_compress_tool_success_count,
            enabled_compress_tool_error_count=int(enabled_diagnostics.get("compress_tool_error_count", 0)),
            enabled_compress_tool_effective=enabled_compress_tool_effective,
        ),
        "compress_tool_helped": enabled_passed and not disabled_passed and enabled_compress_tool_effective,
        "both_passed": enabled_passed and disabled_passed,
        "both_failed": not enabled_passed and not disabled_passed,
        "compress_tool_regressed": not enabled_passed and disabled_passed,
        "enabled_score": enabled_grade["score"],
        "disabled_score": disabled_grade["score"],
        "score_delta": enabled_grade["score"] - disabled_grade["score"],
        "enabled_compress_tool_call_count": enabled_diagnostics.get("compress_tool_call_count", 0),
        "enabled_compress_tool_success_count": enabled_compress_tool_success_count,
        "enabled_compress_tool_error_count": enabled_diagnostics.get("compress_tool_error_count", 0),
        "enabled_compress_tool_effective": enabled_compress_tool_effective,
        "disabled_compress_tool_call_count": disabled_diagnostics.get("compress_tool_call_count", 0),
        "enabled_read_file_tool_result_chars": enabled_diagnostics.get("read_file_tool_result_chars", 0),
        "disabled_read_file_tool_result_chars": disabled_diagnostics.get("read_file_tool_result_chars", 0),
        "enabled_elapsed_seconds": enabled_diagnostics.get("elapsed_seconds", enabled.get("elapsed_seconds", 0)),
        "disabled_elapsed_seconds": disabled_diagnostics.get("elapsed_seconds", disabled.get("elapsed_seconds", 0)),
        "enabled_estimated_tool_context_chars_saved": enabled_saved_chars,
    }


def _summarize_suite(cases: dict[str, dict[str, object]]) -> dict[str, object]:
    comparisons: list[dict[str, object]] = []
    enabled_pass_count = 0
    disabled_pass_count = 0
    enabled_score_total = 0
    disabled_score_total = 0
    enabled_elapsed_seconds = 0.0
    disabled_elapsed_seconds = 0.0
    enabled_compress_tool_effective_count = 0
    enabled_compress_tool_call_count = 0
    enabled_estimated_tool_context_chars_saved = 0

    for pressure_results in cases.values():
        for result in pressure_results.values():
            if not isinstance(result, dict):
                continue
            comparison = result["comparison"]
            enabled = result["enabled"]
            disabled = result["disabled"]
            comparisons.append(comparison)
            if enabled["grade"]["passed"]:
                enabled_pass_count += 1
            if disabled["grade"]["passed"]:
                disabled_pass_count += 1
            enabled_score_total += int(enabled["grade"]["score"])
            disabled_score_total += int(disabled["grade"]["score"])
            enabled_elapsed_seconds += float(enabled.get("elapsed_seconds", 0))
            disabled_elapsed_seconds += float(disabled.get("elapsed_seconds", 0))
            if comparison.get("enabled_compress_tool_effective"):
                enabled_compress_tool_effective_count += 1
            enabled_compress_tool_call_count += int(comparison.get("enabled_compress_tool_call_count", 0))
            enabled_estimated_tool_context_chars_saved += int(
                comparison.get("enabled_estimated_tool_context_chars_saved", 0)
            )

    cell_count = len(comparisons)
    return {
        "cell_count": cell_count,
        "enabled_pass_count": enabled_pass_count,
        "disabled_pass_count": disabled_pass_count,
        "enabled_success_rate": _ratio(enabled_pass_count, cell_count),
        "disabled_success_rate": _ratio(disabled_pass_count, cell_count),
        "correctness_delta": enabled_pass_count - disabled_pass_count,
        "enabled_score_total": enabled_score_total,
        "disabled_score_total": disabled_score_total,
        "score_delta_total": enabled_score_total - disabled_score_total,
        "compress_tool_helped_count": sum(1 for item in comparisons if item.get("compress_tool_helped")),
        "both_passed_count": sum(1 for item in comparisons if item.get("both_passed")),
        "both_failed_count": sum(1 for item in comparisons if item.get("both_failed")),
        "compress_tool_regressed_count": sum(1 for item in comparisons if item.get("compress_tool_regressed")),
        "enabled_compress_tool_effective_count": enabled_compress_tool_effective_count,
        "enabled_compress_tool_call_count": enabled_compress_tool_call_count,
        "enabled_estimated_tool_context_chars_saved": enabled_estimated_tool_context_chars_saved,
        "enabled_elapsed_seconds": round(enabled_elapsed_seconds, 3),
        "disabled_elapsed_seconds": round(disabled_elapsed_seconds, 3),
        "elapsed_seconds_delta": round(enabled_elapsed_seconds - disabled_elapsed_seconds, 3),
    }


def _summarize_policy_results(policy_results: dict[str, dict[str, object]]) -> dict[str, object]:
    policy_summaries = {
        policy: result["summary"]
        for policy, result in policy_results.items()
        if isinstance(result.get("summary"), dict)
    }
    cell_count = sum(int(summary.get("cell_count", 0)) for summary in policy_summaries.values())
    enabled_pass_count = sum(
        int(summary.get("enabled_pass_count", 0)) for summary in policy_summaries.values()
    )
    disabled_pass_count = sum(
        int(summary.get("disabled_pass_count", 0)) for summary in policy_summaries.values()
    )
    enabled_score_total = sum(
        int(summary.get("enabled_score_total", 0)) for summary in policy_summaries.values()
    )
    disabled_score_total = sum(
        int(summary.get("disabled_score_total", 0)) for summary in policy_summaries.values()
    )
    enabled_elapsed_seconds = sum(
        float(summary.get("enabled_elapsed_seconds", 0)) for summary in policy_summaries.values()
    )
    disabled_elapsed_seconds = sum(
        float(summary.get("disabled_elapsed_seconds", 0)) for summary in policy_summaries.values()
    )
    return {
        "policy_count": len(policy_summaries),
        "cell_count": cell_count,
        "enabled_pass_count": enabled_pass_count,
        "disabled_pass_count": disabled_pass_count,
        "enabled_success_rate": _ratio(enabled_pass_count, cell_count),
        "disabled_success_rate": _ratio(disabled_pass_count, cell_count),
        "correctness_delta": enabled_pass_count - disabled_pass_count,
        "enabled_score_total": enabled_score_total,
        "disabled_score_total": disabled_score_total,
        "score_delta_total": enabled_score_total - disabled_score_total,
        "compress_tool_helped_count": sum(
            int(summary.get("compress_tool_helped_count", 0)) for summary in policy_summaries.values()
        ),
        "both_passed_count": sum(
            int(summary.get("both_passed_count", 0)) for summary in policy_summaries.values()
        ),
        "both_failed_count": sum(
            int(summary.get("both_failed_count", 0)) for summary in policy_summaries.values()
        ),
        "compress_tool_regressed_count": sum(
            int(summary.get("compress_tool_regressed_count", 0)) for summary in policy_summaries.values()
        ),
        "enabled_compress_tool_effective_count": sum(
            int(summary.get("enabled_compress_tool_effective_count", 0))
            for summary in policy_summaries.values()
        ),
        "enabled_compress_tool_call_count": sum(
            int(summary.get("enabled_compress_tool_call_count", 0))
            for summary in policy_summaries.values()
        ),
        "enabled_estimated_tool_context_chars_saved": sum(
            int(summary.get("enabled_estimated_tool_context_chars_saved", 0))
            for summary in policy_summaries.values()
        ),
        "enabled_elapsed_seconds": round(enabled_elapsed_seconds, 3),
        "disabled_elapsed_seconds": round(disabled_elapsed_seconds, 3),
        "elapsed_seconds_delta": round(enabled_elapsed_seconds - disabled_elapsed_seconds, 3),
        "pure_availability_delta": _policy_correctness_delta(
            policy_summaries,
            TOOL_POLICY_NEUTRAL,
        ),
        "guided_tool_delta": _policy_correctness_delta(
            policy_summaries,
            TOOL_POLICY_GUIDED,
        ),
        "policy_summaries": policy_summaries,
    }


def _policy_correctness_delta(policy_summaries: dict[str, dict[str, object]], policy: str) -> int | None:
    summary = policy_summaries.get(policy)
    if summary is None:
        return None
    return int(summary.get("correctness_delta", 0))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _comparison_verdict(
    *,
    enabled_passed: bool,
    disabled_passed: bool,
    enabled_compress_tool_call_count: int,
    enabled_compress_tool_success_count: int,
    enabled_compress_tool_error_count: int,
    enabled_compress_tool_effective: bool,
) -> str:
    if enabled_passed and not disabled_passed:
        if enabled_compress_tool_effective:
            return "compress_tool_helped"
        return "enabled_passed_without_successful_compress_tool"
    if not enabled_passed and disabled_passed:
        return "compress_tool_regressed"
    if enabled_passed and disabled_passed:
        if enabled_compress_tool_error_count:
            return "inconclusive_enabled_compress_tool_failed"
        if enabled_compress_tool_call_count == 0:
            return "inconclusive_enabled_did_not_call_compress_tool"
        if enabled_compress_tool_success_count == 0:
            return "inconclusive_enabled_compress_tool_not_effective"
        return "inconclusive_both_passed"
    if enabled_compress_tool_error_count:
        return "both_failed_enabled_compress_tool_failed"
    return "both_failed"


def _flatten_for_line_cli(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\r\n", "\n").replace("\n", "\\n")


def _extract_banner_path(stdout: str, label: str) -> str | None:
    prefix = f"{label}: "
    for line in stdout.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return None


def _read_text_if_present(path_text: str | None) -> str:
    if not path_text:
        return ""
    path = Path(path_text)
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _iter_session_sections(session_text: str) -> Iterator[SessionSection]:
    header_pattern = re.compile(r"^## \d{2}:\d{2}:\d{2} (?P<title>[^\n]+)\n\n", re.MULTILINE)
    search_start = 0
    while True:
        header_match = header_pattern.search(session_text, search_start)
        if header_match is None:
            return

        fence_line_start = header_match.end()
        fence_line_end = session_text.find("\n", fence_line_start)
        if fence_line_end == -1:
            return

        fence_line = session_text[fence_line_start:fence_line_end]
        fence_match = re.fullmatch(r"(?P<fence>`{3,})(?P<language>[A-Za-z0-9_-]*)", fence_line)
        if fence_match is None:
            search_start = fence_line_end + 1
            continue

        fence = fence_match.group("fence")
        close_pattern = re.compile(rf"^{re.escape(fence)}$", re.MULTILINE)
        content_start = fence_line_end + 1
        close_match = close_pattern.search(session_text, content_start)
        if close_match is None:
            return

        yield SessionSection(
            title=header_match.group("title"),
            language=fence_match.group("language"),
            content=session_text[content_start : close_match.start()].rstrip("\n"),
        )
        search_start = close_match.end()


def _tool_result_text_chars(sections: Sequence[SessionSection], tool_name: str) -> int:
    title = f"ToolResult {tool_name}"
    return sum(len(section.content) for section in sections if section.title == title and section.language == "text")


def _tool_result_chars_by_call_signature(
    sections: Sequence[SessionSection],
    *,
    tool_name: str | None = None,
) -> dict[tuple[str, str], list[int]]:
    chars_by_signature: dict[tuple[str, str], list[int]] = {}
    pending_tool_call: tuple[str, str] | None = None
    for section in sections:
        if section.title.startswith("ToolCall ") and section.language == "json":
            parsed = _parse_json_section(section)
            if parsed is None:
                pending_tool_call = None
                continue
            raw_name = parsed.get("name")
            raw_arguments = parsed.get("arguments")
            if isinstance(raw_name, str) and isinstance(raw_arguments, dict):
                pending_tool_call = (raw_name, _canonical_json(raw_arguments))
            else:
                pending_tool_call = None
            continue

        if not section.title.startswith("ToolResult ") or section.language != "text":
            continue
        if pending_tool_call is None:
            continue

        pending_name, pending_arguments_json = pending_tool_call
        result_name = section.title.removeprefix("ToolResult ")
        if pending_name == result_name and (tool_name is None or pending_name == tool_name):
            signature = (pending_name, pending_arguments_json)
            chars_by_signature.setdefault(signature, []).append(len(section.content))
        pending_tool_call = None

    return chars_by_signature


def _compress_replacement_stats(sections: Sequence[SessionSection]) -> dict[str, object]:
    all_tool_result_chars_by_signature = _tool_result_chars_by_call_signature(sections)
    attempted_replacement_chars = 0
    successful_replacement_chars = 0
    estimated_saved_chars = 0
    success_count = 0
    errors: list[str] = []
    pending_replacements: list[tuple[str, str, str]] = []

    for section in sections:
        if section.title == "ToolCall compress_tool" and section.language == "json":
            pending_replacements = _compress_replacements_from_section(section)
            attempted_replacement_chars += sum(
                len(replace_content) for _, _, replace_content in pending_replacements
            )
            continue

        if section.title != "ToolResult compress_tool" or section.language != "text":
            continue

        result_text = section.content.strip()
        if result_text == "Success":
            success_count += 1
            for tool_name, tool_arguments_json, replace_content in pending_replacements:
                successful_replacement_chars += len(replace_content)
                original_chars = all_tool_result_chars_by_signature.get(
                    (tool_name, tool_arguments_json),
                    [],
                )
                if len(original_chars) == 1:
                    estimated_saved_chars += max(0, original_chars[0] - len(replace_content))
        elif result_text.startswith("Error:"):
            errors.append(result_text)
        pending_replacements = []

    return {
        "attempted_replacement_chars": attempted_replacement_chars,
        "successful_replacement_chars": successful_replacement_chars,
        "estimated_tool_context_chars_saved": estimated_saved_chars,
        "success_count": success_count,
        "error_count": len(errors),
        "errors": errors,
    }


def _compress_replacements_from_section(section: SessionSection) -> list[tuple[str, str, str]]:
    data = _parse_json_section(section)
    if data is None:
        return []
    raw_arguments = data.get("arguments")
    if not isinstance(raw_arguments, dict):
        return []
    raw_replacements = raw_arguments.get("replacements", [])
    if not isinstance(raw_replacements, list):
        return []

    replacements: list[tuple[str, str, str]] = []
    for raw_replacement in raw_replacements:
        if not isinstance(raw_replacement, dict):
            continue
        tool_name = raw_replacement.get("tool_name")
        tool_arguments = raw_replacement.get("tool_arguments")
        replace_content = raw_replacement.get("replace_content")
        if (
            isinstance(tool_name, str)
            and isinstance(tool_arguments, dict)
            and isinstance(replace_content, str)
        ):
            replacements.append((_normalize_tool_name(tool_name), _canonical_json(tool_arguments), replace_content))
    return replacements


def _normalize_tool_name(tool_name: str) -> str:
    return tool_name.rsplit(".", 1)[-1]


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _parse_json_section(section: SessionSection) -> dict[str, object] | None:
    try:
        data = json.loads(section.content)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _run_command_with_progress(
    cmd: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    stdin_text: str,
    timeout_seconds: int,
    progress_interval_seconds: float,
    show_progress: bool,
    label: str,
    log_stream: TextIO,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    started_at = time.monotonic()
    result_box: dict[str, object] = {}

    def communicate() -> None:
        try:
            stdout, stderr = process.communicate(input=stdin_text)
            result_box["completed"] = subprocess.CompletedProcess(
                args=cmd,
                returncode=process.returncode,
                stdout=stdout,
                stderr=stderr,
            )
        except BaseException as exc:  # noqa: BLE001 - re-raised in the main thread.
            result_box["exception"] = exc

    worker = threading.Thread(target=communicate, daemon=True)
    worker.start()
    last_rendered_second = -1
    try:
        while worker.is_alive():
            elapsed_seconds = time.monotonic() - started_at
            if elapsed_seconds >= timeout_seconds:
                process.kill()
                worker.join()
                _finish_progress_line(
                    show_progress=show_progress,
                    status=f"{label} 运行超时，已终止",
                    elapsed_seconds=elapsed_seconds,
                    stream=sys.stderr,
                )
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout_seconds)

            rendered_second = int(elapsed_seconds)
            if rendered_second != last_rendered_second:
                if rendered_second == 0 or rendered_second % max(1, int(progress_interval_seconds)) == 0:
                    _write_run_log(log_stream, f"{label} running elapsed={_format_elapsed(elapsed_seconds)}")
                _render_progress_line(
                    show_progress=show_progress,
                    label=label,
                    elapsed_seconds=elapsed_seconds,
                    stream=sys.stderr,
                )
                last_rendered_second = rendered_second
            time.sleep(max(0.1, progress_interval_seconds))
    except KeyboardInterrupt:
        process.kill()
        worker.join()
        _finish_progress_line(
            show_progress=show_progress,
            status=f"{label} 已中断，子进程已终止",
            elapsed_seconds=time.monotonic() - started_at,
            stream=sys.stderr,
        )
        raise

    worker.join()
    elapsed_seconds = time.monotonic() - started_at
    _finish_progress_line(
        show_progress=show_progress,
        status=f"{label} 运行结束",
        elapsed_seconds=elapsed_seconds,
        stream=sys.stderr,
    )
    _write_run_log(
        log_stream,
        f"{label} completed returncode={process.returncode} elapsed={_format_elapsed(elapsed_seconds)}",
    )

    if "exception" in result_box:
        raise result_box["exception"]
    return result_box["completed"]  # type: ignore[return-value]


def _render_progress_line(*, show_progress: bool, label: str, elapsed_seconds: float, stream: TextIO) -> None:
    if not show_progress:
        return
    with _PROGRESS_LOCK:
        stream.write(f"\r\033[K{label} 正在运行中...已运行{_format_elapsed(elapsed_seconds)}")
        stream.flush()


def _finish_progress_line(*, show_progress: bool, status: str, elapsed_seconds: float, stream: TextIO) -> None:
    if not show_progress:
        return
    with _PROGRESS_LOCK:
        stream.write(f"\r\033[K{status}，耗时{_format_elapsed(elapsed_seconds)}\n")
        stream.flush()


def _format_elapsed(elapsed_seconds: float) -> str:
    total_seconds = max(0, int(elapsed_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}小时{minutes}分{seconds}秒"
    if minutes:
        return f"{minutes}分{seconds}秒"
    return f"{seconds}秒"


if __name__ == "__main__":
    raise SystemExit(main())
