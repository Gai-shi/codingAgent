# GitHub README Redesign Spec

## Goal

Rewrite the repository documentation for developers and interviewers while preserving the project's honest positioning as a learning-oriented coding agent.

## Deliverables

- `README.md`: the default English GitHub landing page.
- `README.zh-CN.md`: a complete Simplified Chinese version.
- Language-switch links at the top of both documents.

## Positioning

Use this core description:

> A dependency-free, learning-oriented coding agent built from first principles in Python.

The documentation must demonstrate engineering depth without presenting the project as a production-ready agent framework.

## Information Architecture

Both documents use the same section order:

1. Project title, language switch, concise positioning, and factual badges.
2. Overview and core capabilities.
3. Architecture diagram and execution flow.
4. Quick start and configuration.
5. Built-in tools and workspace safety boundaries.
6. Context management and compression design.
7. Project structure.
8. Testing and real-LLM E2E evaluations.
9. Design principles and known limitations.

## Content Rules

- Describe only behavior verified in the current source tree.
- Keep the opening concise enough for a reader to understand the project within approximately 30 seconds.
- Share code examples, configuration keys, and diagrams conceptually across both language versions.
- Translate ideas naturally instead of mirroring sentences mechanically.
- Prefer tables and diagrams over long development-history lists.
- Avoid unstable claims such as unverified benchmark improvements or production readiness.
- Do not expose local absolute paths, credentials, or machine-specific ModelHub configuration.

## Validation

- Verify every documented CLI flag and environment variable against the source code.
- Verify every listed tool against the default tool registry.
- Verify documented Python compatibility against `pyproject.toml`.
- Run the existing automated test suite after editing.
- Inspect the final Git diff for accidental changes outside the README deliverables.
