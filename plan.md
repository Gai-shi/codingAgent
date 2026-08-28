# Runner Comparison E2E Eval Implementation Plan

## File Map

- `evals/runner_comparison_e2e/__init__.py`: package marker for the new eval.
- `evals/runner_comparison_e2e/benchmark_case.py`: creates target repos and prompt turns for the six runner comparison cases.
- `evals/runner_comparison_e2e/grader.py`: deterministic graders for final assistant text and target repo state.
- `evals/runner_comparison_e2e/run_ai_job_ab.py`: CLI entrypoint that runs native/langgraph variants, collects diagnostics, estimates tokens, and writes summaries.
- `evals/runner_comparison_e2e/README.md`: usage notes, metric definitions, and limitations.

## Tasks

1. Create `evals/runner_comparison_e2e/__init__.py` with a short package docstring.

2. Add case definitions in `evals/runner_comparison_e2e/benchmark_case.py` for `direct_answer`, `read_file_answer`, `grep_then_read`, `single_file_patch`, `multi_file_patch`, and `tool_error_recovery`.

3. Add workspace generation in `evals/runner_comparison_e2e/benchmark_case.py` so each case gets only the files required by that case and a `.ai_job_eval_case.json` manifest.

4. Add prompt generation helpers in `evals/runner_comparison_e2e/benchmark_case.py` so each case can produce one or more CLI prompt turns and prompt statistics.

5. Add deterministic grading in `evals/runner_comparison_e2e/grader.py` for final assistant text cases and file mutation cases.

6. Add small runtime checks in `evals/runner_comparison_e2e/grader.py` for patch cases using local Python subprocesses in the target repo.

7. Add the A/B command runner in `evals/runner_comparison_e2e/run_ai_job_ab.py` that schedules native and langgraph variants for the selected cases.

8. Add per-variant artifact writing in `evals/runner_comparison_e2e/run_ai_job_ab.py` for stdin prompts, stdout, stderr, trace log path, session record path, and result JSON.

9. Add diagnostics and estimated token extraction in `evals/runner_comparison_e2e/run_ai_job_ab.py` from prompt text, final assistant text, and session record sections.

10. Add per-case comparison and aggregate summary in `evals/runner_comparison_e2e/run_ai_job_ab.py` for correctness, score, elapsed seconds, estimated tokens, and ratio fields.

11. Add compact terminal progress and run log writing in `evals/runner_comparison_e2e/run_ai_job_ab.py`.

12. Add `evals/runner_comparison_e2e/README.md` documenting command usage, output layout, estimated token semantics, and the end-to-end comparison limitation.

13. Run syntax and targeted import verification for the new eval package.

14. Run a small no-model-safe grader/fixture verification by creating generated workspaces and checking initial failure plus known passing target edits.

15. Run the existing relevant test subset if available without requiring network or live model calls.
