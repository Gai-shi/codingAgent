# Runner Comparison E2E Eval Spec

## Goal

Build an end-to-end evaluation suite that compares `native` and `langgraph`
runner behavior on the same task set. The eval should answer three questions:

- How much estimated token budget does each runner consume?
- How often does each runner complete the task correctly?
- How much wall-clock time does each runner take?

This is a testing utility. It should be practical, deterministic where possible,
and aligned with the project's learning goal without making the runner
implementations more complex.

## Scope

Create a new eval package:

```text
evals/runner_comparison_e2e/
```

The suite runs the same generated benchmark cases twice:

- `python -m ai_job --runner native`
- `python -m ai_job --runner langgraph`

Each variant uses its own target workspace, trace log, and session record so the
results cannot interfere with each other.

## Non-Goals

- Do not measure provider-level billing tokens in the first version.
- Do not add an LLM judge.
- Do not refactor the core runner abstractions unless the eval reveals a small
  necessary seam.
- Do not reuse the existing compression pressure eval as the case set.

## Token Measurement

Token counts use the project's existing lightweight estimator:

```text
ai_job.agent.token_counting.estimate_text_tokens
```

The eval estimates tokens from persisted session records and generated prompt
text. It reports estimated token values clearly as estimates, not API billing
usage.

The first version should report at least:

- `estimated_prompt_tokens`
- `estimated_assistant_tokens`
- `estimated_tool_result_tokens`
- `estimated_total_tokens`

The comparison summary should include native/langgraph totals and deltas.

## Correctness Measurement

Correctness is deterministic. Each case has a generated target repo plus a
Python grader that checks final files or final assistant text.

The first version includes six cases:

1. `direct_answer`: answer a precise fact without needing tools.
2. `read_file_answer`: read a file and report a hidden value.
3. `grep_then_read`: locate a target file through search, then answer.
4. `single_file_patch`: update one source file so a small behavior check passes.
5. `multi_file_patch`: update two related files so a small integration check
   passes.
6. `tool_error_recovery`: recover from an initially misleading path by searching
   the workspace and using the correct file.

The summary reports:

- pass count per runner
- success rate per runner
- score total per runner
- pass/score deltas

## Timing Measurement

Elapsed time is measured end to end for each variant process, from immediately
before launching the CLI subprocess to immediately after it exits. This includes:

- CLI startup
- model requests
- tool execution
- file writes
- grader-independent exit handling

The comparison summary reports native/langgraph totals, deltas, and a
`langgraph_to_native_latency_ratio`.

## Command-Line Interface

The main entrypoint is:

```bash
python -m evals.runner_comparison_e2e.run_ai_job_ab --output /tmp/runner_eval --force
```

Useful options:

- `--case-id all|<case_id>`
- `--max-workers N`
- `--timeout-seconds N`
- `--ai-job-command "python -m ai_job"`
- `--ai-job-source-root <repo>`
- `--progress` / `--no-progress`

## Output Files

The suite writes:

```text
<output>/
  result_runner_comparison_ab.json
  run_ai_job_ab.log
  <case_id>/
    native/
      target_repo/
      stdin_prompts.txt
      ai_job_stdout.txt
      ai_job_stderr.txt
      result_native.json
    langgraph/
      target_repo/
      stdin_prompts.txt
      ai_job_stdout.txt
      ai_job_stderr.txt
      result_langgraph.json
```

## Result Shape

Each variant result includes:

- runner name
- command
- exit code
- elapsed seconds
- target path
- diagnostics
- token estimates
- grader result

The top-level result includes:

- selected case ids
- per-case native/langgraph results
- per-case comparison
- aggregate summary

## Design Constraint

The eval intentionally compares real CLI behavior. Because `native` uses the
project's `OpenAIModel` adapter and `langgraph` uses `langchain_openai.ChatOpenAI`,
the result is an end-to-end comparison, not a pure framework overhead benchmark.
The output JSON and README should state this limitation explicitly.
