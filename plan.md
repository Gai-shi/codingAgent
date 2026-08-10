# Automatic Context Compression Plan

## File Structure

- `ai_job/compress/__init__.py`: export compression planning types, functions, and `CompressionManager`.
- `ai_job/compress/context_compression.py`: own pure compression planning rules currently in `ai_job/agent/context_compression.py`.
- `ai_job/compress/compression_manager.py`: own threshold checking, plan creation, summarizer invocation, and in-place history rewriting.
- `ai_job/agent/agent_runner.py`: call `CompressionManager` before each model request while preserving the existing agent loop behavior.
- `ai_job/chat_cli.py`: compose `CompressionManager` with the runner and provide the summarizer callback.
- Tests under `tests/`: update existing imports and add focused coverage only where current tests cannot verify the new integration.

## Tasks

1. Move compression planning ownership into `ai_job/compress`.
   - Create `ai_job/compress/__init__.py`.
   - Move the existing pure planning module from `ai_job/agent/context_compression.py` to `ai_job/compress/context_compression.py`.
   - Update existing tests and imports that reference the old module path.
   - Run `python3 -m pytest tests/test_context_compression.py`.
   - Expected result: the context compression planning tests pass.

2. Add `CompressionManager`.
   - Create `ai_job/compress/compression_manager.py`.
   - Add a `Summarizer` callable contract that accepts `CompressionPlan` and `MessageHistory` and returns `SummaryMessage`.
   - Add `CompressionManager.compress_if_needed(history)` with in-place history mutation.
   - Make compression failures propagate without swallowing exceptions.
   - Add focused tests for no-op behavior, in-place rewrite behavior, and summarizer failure propagation.
   - Run `python3 -m pytest tests/test_context_compression.py`.
   - Expected result: planning and manager tests pass.

3. Wire `CompressionManager` into `AgentRunner`.
   - Update `ai_job/agent/agent_runner.py` to accept an optional compression manager.
   - Ensure the manager is invoked before every model request inside the existing tool loop.
   - Preserve existing behavior when no manager is provided.
   - Extend existing runner tests only if current tests do not verify the integration.
   - Run `python3 -m pytest tests/test_agent_runner.py tests/test_context_compression.py`.
   - Expected result: runner behavior remains compatible and compression invocation is verified.

4. Compose runtime summarization in `chat_cli`.
   - Update `ai_job/chat_cli.py` to construct `CompressionManager` with context window, reserve tokens, keep recent tokens, token counter, and summarizer callback.
   - The summarizer callback should use the existing chat model with an empty tool registry to produce a `SummaryMessage`.
   - Keep summary failure behavior as a raised runtime error so the current turn is interrupted and rolled back by existing CLI logic.
   - Run `python3 -m pytest tests/test_chat_cli_workspace.py tests/test_message_and_provider_adapters.py`.
   - Expected result: CLI composition and provider rendering tests pass.

5. Run focused verification.
   - Run `python3 -m pytest tests/test_context_compression.py tests/test_token_counting.py tests/test_agent_runner.py tests/test_message_and_provider_adapters.py tests/test_chat_cli_workspace.py`.
   - Expected result: all focused tests pass.
