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
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

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

    completed = subprocess.run(
        cmd,
        cwd=source_root,
        env=env,
        input=stdin_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.timeout_seconds,
        check=False,
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


if __name__ == "__main__":
    raise SystemExit(main())
