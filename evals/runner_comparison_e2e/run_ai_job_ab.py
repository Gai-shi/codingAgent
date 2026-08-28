"""Run a real ai_job A/B eval for native and LangGraph runners."""

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

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from ai_job.agent.token_counting import estimate_text_tokens

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark_case import (
        CASE_IDS,
        PromptTurn,
        build_prompt_turns,
        case_ids,
        create_case_workspace,
        prompt_stats,
        prompt_texts,
        write_prompt_artifacts,
    )
    from grader import grade_target
else:
    from .benchmark_case import (
        CASE_IDS,
        PromptTurn,
        build_prompt_turns,
        case_ids,
        create_case_workspace,
        prompt_stats,
        prompt_texts,
        write_prompt_artifacts,
    )
    from .grader import grade_target


RUNNER_NATIVE = "native"
RUNNER_LANGGRAPH = "langgraph"
RUNNERS = (RUNNER_NATIVE, RUNNER_LANGGRAPH)
AI_JOB_TRACE_LOG_PATH_ENV = "AI_JOB_TRACE_LOG_PATH"
AI_JOB_SESSION_RECORD_PATH_ENV = "AI_JOB_SESSION_RECORD_PATH"

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
    case_id: str
    runner: str
    turns: Sequence[PromptTurn]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run native vs LangGraph E2E eval against ai_job.")
    parser.add_argument("--output", required=True, help="Benchmark output directory.")
    parser.add_argument("--force", action="store_true", help="Replace output directory if it exists.")
    parser.add_argument("--case-id", choices=tuple(["all", *CASE_IDS]), default="all")
    parser.add_argument(
        "--ai-job-command",
        default=f"{sys.executable} -m ai_job",
        help="Command used to start ai_job, without --workspace or --runner.",
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
        default=1,
        help="Maximum number of ai_job variant processes to run concurrently. Defaults to serial for cleaner timing.",
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

    if args.max_workers < 1:
        parser.error("--max-workers must be >= 1")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be > 0")
    if args.progress_interval_seconds <= 0:
        parser.error("--progress-interval-seconds must be > 0")

    output = Path(args.output).expanduser().resolve()
    if output.exists() and args.force:
        import shutil

        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    run_log_path = output / "run_ai_job_ab.log"

    selected_case_ids = case_ids(args.case_id)
    case_inputs: dict[str, dict[str, object]] = {}
    variant_tasks: list[VariantTask] = []
    for case_id in selected_case_ids:
        case_output = output / case_id
        case_output.mkdir(parents=True, exist_ok=True)
        turns = build_prompt_turns(case_id=case_id)
        write_prompt_artifacts(case_output, turns)
        case_inputs[case_id] = {
            "case_id": case_id,
            "turns": turns,
            "prompt_stats": prompt_stats(turns, case_id=case_id),
        }
        for runner in RUNNERS:
            variant_tasks.append(
                VariantTask(
                    output=case_output,
                    case_id=case_id,
                    runner=runner,
                    turns=turns,
                )
            )

    with run_log_path.open("w", encoding="utf-8") as run_log:
        _write_run_log(run_log, "ai_job runner comparison eval started")
        _write_run_log(run_log, f"output={output}")
        _write_run_log(run_log, f"case_ids={selected_case_ids}")
        _write_run_log(run_log, f"runners={list(RUNNERS)}")
        _write_run_log(run_log, f"max_workers={args.max_workers}")
        variant_results = _run_variant_tasks(args, variant_tasks, run_log)

    cases: dict[str, dict[str, object]] = {}
    for case_id in selected_case_ids:
        native = variant_results[(case_id, RUNNER_NATIVE)]
        langgraph = variant_results[(case_id, RUNNER_LANGGRAPH)]
        cases[case_id] = {
            "case_id": case_id,
            "prompt_stats": case_inputs[case_id]["prompt_stats"],
            RUNNER_NATIVE: native,
            RUNNER_LANGGRAPH: langgraph,
            "comparison": _compare(native, langgraph),
        }

    result = {
        "runner": "ai_job_runner_comparison_suite",
        "case_ids": selected_case_ids,
        "runners": list(RUNNERS),
        "run_log": str(run_log_path),
        "cases": cases,
        "summary": _summarize_suite(cases),
        "measurement_notes": {
            "token_measurement": "estimated with ai_job.agent.token_counting.estimate_text_tokens; not provider billing usage",
            "comparison_scope": "end-to-end CLI comparison; native and langgraph use different provider adapter paths",
            "timing_scope": "subprocess wall-clock time including CLI startup, model calls, tool execution, and exit handling",
        },
    }
    result_path = output / "result_runner_comparison_ab.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _append_result_summary_log(run_log_path, result_path, result)
    print(_format_run_summary(result_path, run_log_path, result))
    return 0 if result["summary"]["all_variants_passed"] else 1


def _run_variant_tasks(
    args: argparse.Namespace,
    tasks: Sequence[VariantTask],
    run_log: TextIO,
) -> dict[tuple[str, str], dict[str, object]]:
    results: dict[tuple[str, str], dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(_run_variant, args, task, run_log): task
            for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            results[(task.case_id, task.runner)] = future.result()
    return results


def _run_variant(
    args: argparse.Namespace,
    task: VariantTask,
    run_log: TextIO,
) -> dict[str, object]:
    variant_dir = task.output / task.runner
    variant_dir.mkdir(parents=True, exist_ok=True)
    target = create_case_workspace(variant_dir, force=True, case_id=task.case_id)

    effective_prompts = prompt_texts(task.turns)
    (variant_dir / "stdin_prompts.txt").write_text(
        "\n\n--- prompt turn ---\n\n".join(effective_prompts) + "\n",
        encoding="utf-8",
    )
    stdin_text = "\n".join([_flatten_for_line_cli(text) for text in effective_prompts] + ["exit"]) + "\n"
    cmd = shlex.split(args.ai_job_command) + [
        "--workspace",
        str(target),
        "--runner",
        task.runner,
    ]

    env = os.environ.copy()
    source_root = str(Path(args.ai_job_source_root).expanduser().resolve())
    trace_log_base_path = variant_dir / "ai_job_run" / "logs" / "log.log"
    session_record_base_path = variant_dir / "ai_job_run" / "sessions" / "sessions.md"
    env["PYTHONPATH"] = source_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("FILTER_TERMINAL_LOG_LEVEL", "none")
    env[AI_JOB_TRACE_LOG_PATH_ENV] = str(trace_log_base_path)
    env[AI_JOB_SESSION_RECORD_PATH_ENV] = str(session_record_base_path)

    started_at = time.monotonic()
    label = f"{task.case_id}/{task.runner}"
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
    diagnostics = collect_run_diagnostics(
        completed.stdout,
        completed.stderr,
        effective_prompts,
    )
    diagnostics["elapsed_seconds"] = elapsed_seconds
    final_answer = str(diagnostics.get("final_answer", ""))
    grade = grade_target(
        target,
        case_id=task.case_id,
        final_answer=final_answer,
    )
    variant_result = {
        "case_id": task.case_id,
        "runner": task.runner,
        "command": cmd,
        "exit_code": completed.returncode,
        "elapsed_seconds": elapsed_seconds,
        "target": str(target),
        "trace_log_base_path": str(trace_log_base_path),
        "session_record_base_path": str(session_record_base_path),
        "diagnostics": diagnostics,
        "token_estimates": diagnostics["token_estimates"],
        "grade": asdict(grade),
    }
    (variant_dir / f"result_{task.runner}.json").write_text(
        json.dumps(variant_result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_run_log(
        run_log,
        (
            f"{label} graded passed={grade.passed} score={grade.score} "
            f"elapsed_seconds={elapsed_seconds} "
            f"estimated_total_tokens={variant_result['token_estimates']['estimated_total_tokens']}"
        ),
    )
    return variant_result


def collect_run_diagnostics(
    stdout: str,
    stderr: str,
    prompt_text_values: Sequence[str],
) -> dict[str, object]:
    session_record = _extract_banner_path(stdout, "session_record")
    log_file = _extract_banner_path(stdout, "log_file")
    session_text = _read_text_if_present(session_record)
    sections = list(_iter_session_sections(session_text))
    token_estimates = _estimate_tokens(sections, prompt_text_values)
    final_answer = _extract_final_assistant_text(sections) or _extract_final_assistant_text_from_stdout(stdout)
    return {
        "log_file": log_file,
        "session_record": session_record,
        "session_record_chars": len(session_text),
        "startup_failed_count": stderr.count("启动失败："),
        "langgraph_dependency_error_count": stderr.count("请先安装 LangGraph 可选依赖"),
        "llm_request_failed_count": stderr.count("LLM 请求失败"),
        "context_length_exceeded_count": stderr.count("context_length_exceeded"),
        "model_permission_error_count": stderr.count("no model permission"),
        "python_traceback_count": stderr.count("Traceback (most recent call last):"),
        "user_message_count": len([section for section in sections if section.title == "UserMessage"]),
        "assistant_message_count": len([section for section in sections if section.title == "AssistantMessage"]),
        "tool_call_count": len([section for section in sections if section.title.startswith("ToolCall ")]),
        "read_file_tool_call_count": len([section for section in sections if section.title == "ToolCall read_file"]),
        "grep_tool_call_count": len([section for section in sections if section.title == "ToolCall grep"]),
        "apply_patch_tool_call_count": len([section for section in sections if section.title == "ToolCall apply_patch"]),
        "tool_result_count": len([section for section in sections if section.title.startswith("ToolResult ")]),
        "tool_error_count": len(re.findall(r"\nError: ", session_text)),
        "final_answer": final_answer,
        "token_estimates": token_estimates,
    }


def _estimate_tokens(
    sections: Sequence[SessionSection],
    prompt_text_values: Sequence[str],
) -> dict[str, int]:
    current_history_tokens: list[int] = []
    estimated_prompt_tokens = 0
    estimated_assistant_tokens = 0
    estimated_tool_result_tokens = 0
    estimated_user_tokens = 0
    estimated_system_tokens = 0

    for section in sections:
        if section.title == "SystemMessage" and section.language == "text":
            token_count = estimate_text_tokens(section.content)
            estimated_system_tokens += token_count
            current_history_tokens.append(token_count)
            continue
        if section.title == "UserMessage" and section.language == "text":
            token_count = estimate_text_tokens(section.content)
            estimated_user_tokens += token_count
            current_history_tokens.append(token_count)
            continue
        if section.title == "AssistantMessage" and section.language == "json":
            estimated_prompt_tokens += sum(current_history_tokens)
            assistant_tokens = _assistant_section_tokens(section)
            estimated_assistant_tokens += assistant_tokens
            current_history_tokens.append(assistant_tokens)
            continue
        if section.title.startswith("ToolResult ") and section.language == "text":
            tool_tokens = estimate_text_tokens(section.content)
            estimated_tool_result_tokens += tool_tokens
            current_history_tokens.append(tool_tokens)

    generated_prompt_tokens = sum(estimate_text_tokens(text) for text in prompt_text_values)
    estimated_total_tokens = estimated_prompt_tokens + estimated_assistant_tokens
    return {
        "estimated_generated_prompt_tokens": generated_prompt_tokens,
        "estimated_system_tokens": estimated_system_tokens,
        "estimated_user_tokens": estimated_user_tokens,
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "estimated_assistant_tokens": estimated_assistant_tokens,
        "estimated_tool_result_tokens": estimated_tool_result_tokens,
        "estimated_total_tokens": estimated_total_tokens,
    }


def _assistant_section_tokens(section: SessionSection) -> int:
    data = _parse_json_section(section)
    if data is None:
        return estimate_text_tokens(section.content)

    parts: list[str] = []
    content = data.get("content")
    if isinstance(content, str):
        parts.append(content)
    tool_calls = data.get("tool_calls")
    if isinstance(tool_calls, list):
        for raw_tool_call in tool_calls:
            if not isinstance(raw_tool_call, dict):
                continue
            name = raw_tool_call.get("name")
            arguments = raw_tool_call.get("arguments")
            if isinstance(name, str):
                parts.append(name)
            if isinstance(arguments, dict):
                parts.append(_canonical_json(arguments))
    return estimate_text_tokens("\n".join(parts))


def _extract_final_assistant_text(sections: Sequence[SessionSection]) -> str:
    for section in reversed(sections):
        if section.title != "AssistantMessage" or section.language != "json":
            continue
        data = _parse_json_section(section)
        if data is None:
            continue
        content = data.get("content")
        if isinstance(content, str):
            return content
    return ""


def _extract_final_assistant_text_from_stdout(stdout: str) -> str:
    marker = "助手> "
    start = stdout.rfind(marker)
    if start == -1:
        return ""
    text = stdout[start + len(marker):]
    end_match = re.search(r"\n(?:你> |再见。)", text)
    if end_match:
        text = text[: end_match.start()]
    return text.strip()


def _compare(native: dict[str, object], langgraph: dict[str, object]) -> dict[str, object]:
    native_grade = native["grade"]
    langgraph_grade = langgraph["grade"]
    native_tokens = native["token_estimates"]
    langgraph_tokens = langgraph["token_estimates"]
    native_elapsed_seconds = float(native.get("elapsed_seconds", 0))
    langgraph_elapsed_seconds = float(langgraph.get("elapsed_seconds", 0))
    native_passed = bool(native_grade["passed"])
    langgraph_passed = bool(langgraph_grade["passed"])
    return {
        "verdict": _comparison_verdict(native_passed=native_passed, langgraph_passed=langgraph_passed),
        "native_passed": native_passed,
        "langgraph_passed": langgraph_passed,
        "both_passed": native_passed and langgraph_passed,
        "both_failed": not native_passed and not langgraph_passed,
        "native_score": native_grade["score"],
        "langgraph_score": langgraph_grade["score"],
        "langgraph_score_delta": int(langgraph_grade["score"]) - int(native_grade["score"]),
        "native_elapsed_seconds": native_elapsed_seconds,
        "langgraph_elapsed_seconds": langgraph_elapsed_seconds,
        "elapsed_seconds_delta": round(langgraph_elapsed_seconds - native_elapsed_seconds, 3),
        "langgraph_to_native_latency_ratio": _float_ratio(langgraph_elapsed_seconds, native_elapsed_seconds),
        "native_estimated_total_tokens": native_tokens["estimated_total_tokens"],
        "langgraph_estimated_total_tokens": langgraph_tokens["estimated_total_tokens"],
        "estimated_total_tokens_delta": int(langgraph_tokens["estimated_total_tokens"])
        - int(native_tokens["estimated_total_tokens"]),
        "langgraph_to_native_token_ratio": _int_ratio(
            int(langgraph_tokens["estimated_total_tokens"]),
            int(native_tokens["estimated_total_tokens"]),
        ),
    }


def _comparison_verdict(*, native_passed: bool, langgraph_passed: bool) -> str:
    if native_passed and langgraph_passed:
        return "both_passed"
    if native_passed and not langgraph_passed:
        return "native_only_passed"
    if not native_passed and langgraph_passed:
        return "langgraph_only_passed"
    return "both_failed"


def _summarize_suite(cases: dict[str, dict[str, object]]) -> dict[str, object]:
    native_pass_count = 0
    langgraph_pass_count = 0
    native_score_total = 0
    langgraph_score_total = 0
    native_elapsed_seconds = 0.0
    langgraph_elapsed_seconds = 0.0
    native_total_tokens = 0
    langgraph_total_tokens = 0

    for result in cases.values():
        native = result[RUNNER_NATIVE]
        langgraph = result[RUNNER_LANGGRAPH]
        native_grade = native["grade"]
        langgraph_grade = langgraph["grade"]
        if native_grade["passed"]:
            native_pass_count += 1
        if langgraph_grade["passed"]:
            langgraph_pass_count += 1
        native_score_total += int(native_grade["score"])
        langgraph_score_total += int(langgraph_grade["score"])
        native_elapsed_seconds += float(native.get("elapsed_seconds", 0))
        langgraph_elapsed_seconds += float(langgraph.get("elapsed_seconds", 0))
        native_total_tokens += int(native["token_estimates"]["estimated_total_tokens"])
        langgraph_total_tokens += int(langgraph["token_estimates"]["estimated_total_tokens"])

    cell_count = len(cases)
    return {
        "case_count": cell_count,
        "native_pass_count": native_pass_count,
        "langgraph_pass_count": langgraph_pass_count,
        "native_success_rate": _ratio(native_pass_count, cell_count),
        "langgraph_success_rate": _ratio(langgraph_pass_count, cell_count),
        "langgraph_correctness_delta": langgraph_pass_count - native_pass_count,
        "native_score_total": native_score_total,
        "langgraph_score_total": langgraph_score_total,
        "langgraph_score_delta_total": langgraph_score_total - native_score_total,
        "native_elapsed_seconds": round(native_elapsed_seconds, 3),
        "langgraph_elapsed_seconds": round(langgraph_elapsed_seconds, 3),
        "elapsed_seconds_delta": round(langgraph_elapsed_seconds - native_elapsed_seconds, 3),
        "langgraph_to_native_latency_ratio": _float_ratio(langgraph_elapsed_seconds, native_elapsed_seconds),
        "native_estimated_total_tokens": native_total_tokens,
        "langgraph_estimated_total_tokens": langgraph_total_tokens,
        "estimated_total_tokens_delta": langgraph_total_tokens - native_total_tokens,
        "langgraph_to_native_token_ratio": _int_ratio(langgraph_total_tokens, native_total_tokens),
        "both_passed_count": sum(1 for result in cases.values() if result["comparison"]["both_passed"]),
        "both_failed_count": sum(1 for result in cases.values() if result["comparison"]["both_failed"]),
        "native_only_passed_count": sum(
            1 for result in cases.values() if result["comparison"]["verdict"] == "native_only_passed"
        ),
        "langgraph_only_passed_count": sum(
            1 for result in cases.values() if result["comparison"]["verdict"] == "langgraph_only_passed"
        ),
        "all_variants_passed": native_pass_count == cell_count and langgraph_pass_count == cell_count,
    }


def _format_run_summary(result_path: Path, run_log_path: Path, result: dict[str, object]) -> str:
    summary = result["summary"]
    return "\n".join(
        [
            "runner comparison eval completed",
            f"result_json: {result_path}",
            f"run_log: {run_log_path}",
            (
                "summary: "
                f"native_success_rate={summary['native_success_rate']}, "
                f"langgraph_success_rate={summary['langgraph_success_rate']}, "
                f"langgraph_correctness_delta={summary['langgraph_correctness_delta']}, "
                f"native_estimated_total_tokens={summary['native_estimated_total_tokens']}, "
                f"langgraph_estimated_total_tokens={summary['langgraph_estimated_total_tokens']}, "
                f"estimated_total_tokens_delta={summary['estimated_total_tokens_delta']}, "
                f"native_elapsed_seconds={summary['native_elapsed_seconds']}, "
                f"langgraph_elapsed_seconds={summary['langgraph_elapsed_seconds']}, "
                f"elapsed_seconds_delta={summary['elapsed_seconds_delta']}"
            ),
        ]
    )


def _append_result_summary_log(path: Path, result_path: Path, result: dict[str, object]) -> None:
    summary = result["summary"]
    with path.open("a", encoding="utf-8") as stream:
        _write_run_log(stream, f"result={result_path}")
        _write_run_log(
            stream,
            (
                "summary "
                f"native_success_rate={summary['native_success_rate']} "
                f"langgraph_success_rate={summary['langgraph_success_rate']} "
                f"langgraph_correctness_delta={summary['langgraph_correctness_delta']} "
                f"estimated_total_tokens_delta={summary['estimated_total_tokens_delta']} "
                f"elapsed_seconds_delta={summary['elapsed_seconds_delta']}"
            ),
        )


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


def _parse_json_section(section: SessionSection) -> dict[str, object] | None:
    try:
        data = json.loads(section.content)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _int_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _float_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


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
    last_progress_at = -progress_interval_seconds
    try:
        while worker.is_alive():
            elapsed_seconds = time.monotonic() - started_at
            if elapsed_seconds >= timeout_seconds:
                process.kill()
                worker.join()
                _finish_progress_line(
                    show_progress=show_progress,
                    status=f"{label} timed out and was terminated",
                    elapsed_seconds=elapsed_seconds,
                    stream=sys.stderr,
                )
                stdout = ""
                stderr = f"TimeoutExpired: command exceeded {timeout_seconds} seconds"
                completed = result_box.get("completed")
                if isinstance(completed, subprocess.CompletedProcess):
                    stdout = completed.stdout
                    stderr = completed.stderr + ("\n" if completed.stderr else "") + stderr
                _write_run_log(log_stream, f"{label} timeout elapsed={_format_elapsed(elapsed_seconds)}")
                return subprocess.CompletedProcess(args=cmd, returncode=-9, stdout=stdout, stderr=stderr)

            if elapsed_seconds - last_progress_at >= progress_interval_seconds:
                _write_run_log(log_stream, f"{label} running elapsed={_format_elapsed(elapsed_seconds)}")
                _render_progress_line(
                    show_progress=show_progress,
                    label=label,
                    elapsed_seconds=elapsed_seconds,
                    stream=sys.stderr,
                )
                last_progress_at = elapsed_seconds
            worker.join(timeout=0.1)
    except KeyboardInterrupt:
        process.kill()
        worker.join()
        _finish_progress_line(
            show_progress=show_progress,
            status=f"{label} interrupted and terminated",
            elapsed_seconds=time.monotonic() - started_at,
            stream=sys.stderr,
        )
        raise

    worker.join()
    elapsed_seconds = time.monotonic() - started_at
    _finish_progress_line(
        show_progress=show_progress,
        status=f"{label} completed",
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


def _write_run_log(stream: TextIO, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with _RUN_LOG_LOCK:
        stream.write(f"[{timestamp}] {message}\n")
        stream.flush()


def _render_progress_line(*, show_progress: bool, label: str, elapsed_seconds: float, stream: TextIO) -> None:
    if not show_progress:
        return
    with _PROGRESS_LOCK:
        stream.write(f"\r\033[K{label} running... elapsed={_format_elapsed(elapsed_seconds)}")
        stream.flush()


def _finish_progress_line(*, show_progress: bool, status: str, elapsed_seconds: float, stream: TextIO) -> None:
    if not show_progress:
        return
    with _PROGRESS_LOCK:
        stream.write(f"\r\033[K{status}, elapsed={_format_elapsed(elapsed_seconds)}\n")
        stream.flush()


def _format_elapsed(elapsed_seconds: float) -> str:
    total_seconds = max(0, int(elapsed_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes}m{seconds}s"
    if minutes:
        return f"{minutes}m{seconds}s"
    return f"{seconds}s"


if __name__ == "__main__":
    raise SystemExit(main())
