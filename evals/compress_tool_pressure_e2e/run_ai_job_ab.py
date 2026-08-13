"""Run a real ai_job A/B pressure eval for compress_tool."""

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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence, TextIO

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark_case import (
        build_prompt_turns,
        create_case_workspace,
        prompt_stats,
        prompt_texts,
        write_prompt_artifacts,
    )
    from grader import grade_target
else:
    from .benchmark_case import (
        build_prompt_turns,
        create_case_workspace,
        prompt_stats,
        prompt_texts,
        write_prompt_artifacts,
    )
    from .grader import grade_target


@dataclass(frozen=True)
class SessionSection:
    title: str
    language: str
    content: str


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run compress_tool pressure A/B eval against ai_job.")
    parser.add_argument("--output", required=True, help="Benchmark output directory.")
    parser.add_argument("--force", action="store_true", help="Replace output directory if it exists.")
    parser.add_argument("--noise-blocks", type=int, default=420)
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
        "--auto-compression-context-window",
        type=int,
        default=10_000_000,
        help=(
            "Temporary AI_JOB_CONTEXT_WINDOW used during this eval so automatic "
            "context compression does not mask compress_tool behavior."
        ),
    )
    parser.add_argument("--progress-interval-seconds", type=float, default=1.0)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)

    output = Path(args.output).expanduser().resolve()
    if output.exists() and args.force:
        import shutil

        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    turns = build_prompt_turns()
    write_prompt_artifacts(output, turns)

    enabled = _run_variant(args, output=output, turns=turns, variant="enabled", disable_compress_tool=False)
    disabled = _run_variant(args, output=output, turns=turns, variant="disabled", disable_compress_tool=True)
    comparison = _compare(enabled, disabled)
    result = {
        "runner": "ai_job_compress_tool_ab",
        "noise_blocks": args.noise_blocks,
        "prompt_stats": prompt_stats(turns, noise_blocks=args.noise_blocks),
        "enabled": enabled,
        "disabled": disabled,
        "comparison": comparison,
    }
    (output / "result_compress_tool_ab.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if comparison["compress_tool_helped"] else 1


def _run_variant(
    args: argparse.Namespace,
    *,
    output: Path,
    turns: Sequence[object],
    variant: str,
    disable_compress_tool: bool,
) -> dict[str, object]:
    variant_dir = output / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    target = create_case_workspace(variant_dir, force=True, noise_blocks=args.noise_blocks)

    prompts = [_flatten_for_line_cli(text) for text in prompt_texts(turns)]
    stdin_text = "\n".join(prompts + ["exit"]) + "\n"
    cmd = shlex.split(args.ai_job_command) + ["--workspace", str(target)]
    if disable_compress_tool:
        cmd.append("--disable-compress-tool")

    env = os.environ.copy()
    source_root = str(Path(args.ai_job_source_root).expanduser().resolve())
    env["PYTHONPATH"] = source_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["AI_JOB_CONTEXT_WINDOW"] = str(args.auto_compression_context_window)

    completed = _run_command_with_progress(
        cmd,
        cwd=source_root,
        env=env,
        stdin_text=stdin_text,
        timeout_seconds=args.timeout_seconds,
        progress_interval_seconds=args.progress_interval_seconds,
        show_progress=not args.no_progress,
        label=variant,
    )
    (variant_dir / "ai_job_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (variant_dir / "ai_job_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    diagnostics = collect_run_diagnostics(completed.stdout, completed.stderr)
    grade = grade_target(target)
    variant_result = {
        "variant": variant,
        "disable_compress_tool": disable_compress_tool,
        "command": cmd,
        "exit_code": completed.returncode,
        "target": str(target),
        "diagnostics": diagnostics,
        "grade": asdict(grade),
    }
    (variant_dir / f"result_{variant}.json").write_text(
        json.dumps(variant_result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return variant_result


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
        "final_task_user_message_count": session_text.count("请完成当前 workspace 里的 auditor 报告实现"),
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
        "enabled_compress_tool_call_count": enabled_diagnostics["compress_tool_call_count"],
        "enabled_compress_tool_success_count": enabled_compress_tool_success_count,
        "enabled_compress_tool_error_count": enabled_diagnostics.get("compress_tool_error_count", 0),
        "enabled_compress_tool_effective": enabled_compress_tool_effective,
        "disabled_compress_tool_call_count": disabled_diagnostics["compress_tool_call_count"],
        "enabled_read_file_tool_result_chars": enabled_diagnostics.get("read_file_tool_result_chars", 0),
        "disabled_read_file_tool_result_chars": disabled_diagnostics.get("read_file_tool_result_chars", 0),
        "enabled_estimated_tool_context_chars_saved": enabled_saved_chars,
    }


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
            replacements.append((tool_name, _canonical_json(tool_arguments), replace_content))
    return replacements


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
    if "exception" in result_box:
        raise result_box["exception"]  # type: ignore[misc]

    completed = result_box.get("completed")
    if not isinstance(completed, subprocess.CompletedProcess):
        raise RuntimeError("ai_job subprocess finished without a captured result")
    return completed


def _render_progress_line(*, show_progress: bool, label: str, elapsed_seconds: float, stream: TextIO) -> None:
    if not show_progress:
        return
    stream.write(f"\r\033[K{label} 正在运行中...已运行{_format_elapsed(elapsed_seconds)}")
    stream.flush()


def _finish_progress_line(*, show_progress: bool, status: str, elapsed_seconds: float, stream: TextIO) -> None:
    if not show_progress:
        return
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
