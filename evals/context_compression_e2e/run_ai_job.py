"""Run the context compression E2E case against ai_job.

This runner intentionally does not call any compaction command. It is meant to be
used both before and after ai_job gains context management:

- current ai_job is expected to fail this benchmark;
- fixed ai_job with compression should pass the same benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import TextIO, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark_case import (
        build_prompt_turns,
        create_case_workspace,
        prompt_stats,
        prompt_texts_for_ai_job,
        resolve_noise_rounds_for_min_raw_history_chars,
        write_prompt_artifacts,
    )
    from grader import grade_target
else:
    from .benchmark_case import (
        build_prompt_turns,
        create_case_workspace,
        prompt_stats,
        prompt_texts_for_ai_job,
        resolve_noise_rounds_for_min_raw_history_chars,
        write_prompt_artifacts,
    )
    from .grader import grade_target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run context compression E2E case against ai_job.")
    parser.add_argument("--output", required=True, help="Benchmark output directory.")
    parser.add_argument("--force", action="store_true", help="Replace output directory if it exists.")
    parser.add_argument("--noise-rounds", type=int, default=8)
    parser.add_argument("--noise-blocks-per-round", type=int, default=128)
    parser.add_argument(
        "--min-raw-history-chars",
        type=int,
        default=None,
        help=(
            "Increase generated noise until the uncompressed ai_job user-history "
            "has at least this many characters. This uses the real provider "
            "context limit; it does not clip or fake model-visible context."
        ),
    )
    parser.add_argument("--compact-every", type=int, default=4, help="Insert compact turn after every N non-compact turns. ai_job runner skips compact turns.")
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
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=1.0,
        help="Refresh the single-line running status every N seconds.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the single-line running status.",
    )
    args = parser.parse_args(argv)

    output = Path(args.output).expanduser().resolve()
    if output.exists() and args.force:
        import shutil

        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    target = create_case_workspace(output, force=True)
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
    write_prompt_artifacts(output, turns, include_compact_turns=False)

    prompts = [_flatten_for_line_cli(text) for text in prompt_texts_for_ai_job(turns)]
    stdin_text = "\n".join(prompts + ["exit"]) + "\n"

    cmd = shlex.split(args.ai_job_command) + ["--workspace", str(target)]
    env = os.environ.copy()
    source_root = str(Path(args.ai_job_source_root).expanduser().resolve())
    env["PYTHONPATH"] = source_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    completed = _run_command_with_progress(
        cmd,
        cwd=source_root,
        env=env,
        stdin_text=stdin_text,
        timeout_seconds=args.timeout_seconds,
        progress_interval_seconds=args.progress_interval_seconds,
        show_progress=not args.no_progress,
    )

    (output / "ai_job_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output / "ai_job_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    grade = grade_target(target)
    result = {
        "runner": "ai_job",
        "command": cmd,
        "exit_code": completed.returncode,
        "target": str(target),
        "noise_rounds": noise_rounds,
        "noise_blocks_per_round": args.noise_blocks_per_round,
        "prompt_stats": prompt_stats(turns),
        "grade": asdict(grade),
    }
    (output / "result_ai_job.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if grade.passed else 1


def _flatten_for_line_cli(text: str) -> str:
    """ai_job currently reads one prompt with input(), so each turn must be one line."""
    return text.replace("\\", "\\\\").replace("\r\n", "\n").replace("\n", "\\n")


def _run_command_with_progress(
    cmd: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    stdin_text: str,
    timeout_seconds: int,
    progress_interval_seconds: float,
    show_progress: bool,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess while refreshing one terminal status line in-place."""
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
                    status="运行超时，已终止",
                    elapsed_seconds=elapsed_seconds,
                    stream=sys.stderr,
                )
                completed = result_box.get("completed")
                if isinstance(completed, subprocess.CompletedProcess):
                    raise subprocess.TimeoutExpired(
                        cmd=cmd,
                        timeout=timeout_seconds,
                        output=completed.stdout,
                        stderr=completed.stderr,
                    )
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout_seconds)

            rendered_second = int(elapsed_seconds)
            if rendered_second != last_rendered_second:
                _render_progress_line(
                    show_progress=show_progress,
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
            status="已中断，子进程已终止",
            elapsed_seconds=time.monotonic() - started_at,
            stream=sys.stderr,
        )
        raise

    worker.join()
    elapsed_seconds = time.monotonic() - started_at
    _finish_progress_line(
        show_progress=show_progress,
        status="运行结束",
        elapsed_seconds=elapsed_seconds,
        stream=sys.stderr,
    )
    if "exception" in result_box:
        raise result_box["exception"]  # type: ignore[misc]

    completed = result_box.get("completed")
    if not isinstance(completed, subprocess.CompletedProcess):
        raise RuntimeError("ai_job subprocess finished without a captured result")
    return completed


def _render_progress_line(*, show_progress: bool, elapsed_seconds: float, stream: TextIO) -> None:
    if not show_progress:
        return
    stream.write(f"\r\033[K正在运行中...已运行{_format_elapsed(elapsed_seconds)}")
    stream.flush()


def _finish_progress_line(
    *,
    show_progress: bool,
    status: str,
    elapsed_seconds: float,
    stream: TextIO,
) -> None:
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
