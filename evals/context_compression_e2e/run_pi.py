"""Run the context compression E2E case against pi with explicit compaction."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_case import build_prompt_turns, create_case_workspace, prompt_texts_for_pi, write_prompt_artifacts
from grader import grade_target


DEFAULT_PI_COMMAND = "/Users/bytedance/Documents/AI_Projects/storage/pi/pi-test.sh"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run context compression E2E case against pi.")
    parser.add_argument("--output", required=True, help="Benchmark output directory.")
    parser.add_argument("--force", action="store_true", help="Replace output directory if it exists.")
    parser.add_argument("--noise-rounds", type=int, default=8)
    parser.add_argument("--noise-blocks-per-round", type=int, default=96)
    parser.add_argument("--pi-command", default=DEFAULT_PI_COMMAND, help="pi command, e.g. /path/to/pi-test.sh.")
    parser.add_argument("--provider", default=None, help="Optional pi provider.")
    parser.add_argument("--model", default=None, help="Optional pi model.")
    parser.add_argument("--api-key", default=None, help="Optional API key passed to pi.")
    parser.add_argument("--turn-timeout-seconds", type=int, default=1200)
    parser.add_argument(
        "--pi-extra-arg",
        action="append",
        default=[],
        help="Additional arg passed to every pi invocation. Repeat for multiple args.",
    )
    args = parser.parse_args(argv)

    output = Path(args.output).expanduser().resolve()
    if output.exists() and args.force:
        import shutil

        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    target = create_case_workspace(output, force=True)
    turns = build_prompt_turns(
        noise_rounds=args.noise_rounds,
        noise_blocks_per_round=args.noise_blocks_per_round,
    )
    write_prompt_artifacts(output, turns, include_compact_turns=True)

    session_dir = output / "pi_sessions"
    session_dir.mkdir()
    session_id = f"ctx-compress-{int(time.time())}"
    extension_path = Path(__file__).with_name("pi_bench_compact.ts").resolve()

    turn_results: list[dict[str, object]] = []
    env = os.environ.copy()
    prompts = prompt_texts_for_pi(turns)

    for index, prompt in enumerate(prompts, start=1):
        kind = turns[index - 1].kind
        cmd = _build_pi_command(
            args,
            session_dir=session_dir,
            session_id=session_id,
            extension_path=extension_path,
        )
        completed = subprocess.run(
            cmd,
            cwd=target,
            env=env,
            input=prompt,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.turn_timeout_seconds,
            check=False,
        )
        stdout_path = output / f"pi_turn_{index:03d}_{kind}_stdout.txt"
        stderr_path = output / f"pi_turn_{index:03d}_{kind}_stderr.txt"
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        turn_result = {
            "index": index,
            "kind": kind,
            "exit_code": completed.returncode,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        turn_results.append(turn_result)
        if completed.returncode != 0:
            break

    grade = grade_target(target)
    result = {
        "runner": "pi",
        "session_id": session_id,
        "session_dir": str(session_dir),
        "target": str(target),
        "turns": turn_results,
        "grade": asdict(grade),
    }
    (output / "result_pi.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if grade.passed else 1


def _build_pi_command(
    args: argparse.Namespace,
    *,
    session_dir: Path,
    session_id: str,
    extension_path: Path,
) -> list[str]:
    cmd = shlex.split(args.pi_command)
    cmd.extend(
        [
            "-p",
            "--session-id",
            session_id,
            "--session-dir",
            str(session_dir),
            "--extension",
            str(extension_path),
            "--approve",
            "--no-context-files",
            "--no-skills",
            "--no-prompt-templates",
            "--no-themes",
        ]
    )
    if args.provider:
        cmd.extend(["--provider", args.provider])
    if args.model:
        cmd.extend(["--model", args.model])
    if args.api_key:
        cmd.extend(["--api-key", args.api_key])
    cmd.extend(args.pi_extra_arg)
    return cmd


if __name__ == "__main__":
    raise SystemExit(main())
