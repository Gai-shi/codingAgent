# Compress Tool Pressure E2E

这个 eval 用真实 LLM 跑 `ai_job`，比较启用/禁用 `compress_tool` 时的表现。

## 评测目标

压力源来自工具输出，而不是用户 prompt：

1. 目标仓库包含一个很长的 `evidence/noisy_audit_log.txt`。
2. 用户 prompt 要求 agent 先读取该文件。
3. 文件里夹着少量 `FINAL_CONTRACT`，其余是大量废弃值。
4. 启用 `compress_tool` 时，agent 应把 `read_file` 的长输出压缩成少量关键事实，再继续实现。
5. 禁用 `compress_tool` 时，同一个任务会保留完整工具输出，作为 A/B 对照。

runner 会临时设置较大的 `AI_JOB_CONTEXT_WINDOW`，避免自动上下文压缩掩盖 `compress_tool` 的影响。

## 运行

从仓库根目录运行：

```bash
python3 evals/compress_tool_pressure_e2e/run_ai_job_ab.py \
  --output /tmp/ai_job_compress_tool_pressure \
  --force \
  --noise-blocks 420
```

如果想快速冒烟，可以降低噪声：

```bash
python3 evals/compress_tool_pressure_e2e/run_ai_job_ab.py \
  --output /tmp/ai_job_compress_tool_pressure_smoke \
  --force \
  --noise-blocks 20 \
  --no-progress
```

## 结果解释

输出文件：

```text
result_compress_tool_ab.json
enabled/result_enabled.json
disabled/result_disabled.json
enabled/target_repo/
disabled/target_repo/
```

核心字段：

- `comparison.verdict`: 本次 A/B 的可读结论，例如 `compress_tool_helped`、`inconclusive_both_passed`、`inconclusive_enabled_compress_tool_failed`。
- `comparison.compress_tool_helped`: 启用版通过、禁用版失败，且启用版成功压缩并产生上下文节省。
- `comparison.both_passed`: 两边都通过，说明该压力还没拉开差距。
- `comparison.compress_tool_regressed`: 启用版失败、禁用版通过。
- `enabled.diagnostics.compress_tool_call_count`: 模型是否真的调用了 `compress_tool`。
- `enabled.diagnostics.compress_tool_success_count`: `compress_tool` 是否真的执行成功。
- `enabled.diagnostics.compress_tool_errors`: `compress_tool` 失败时的错误文本。
- `enabled.diagnostics.largest_read_file_tool_result_chars`: 单次最大 `read_file` 工具输出字符数。
- `enabled.diagnostics.estimated_tool_context_chars_saved`: 粗略估算的工具输出字符节省量。

如果 `enabled` 没有调用 `compress_tool`，或调用了但 `compress_tool_success_count` 为 0，本次结果不能证明工具没价值，只能说明模型没有按预期成功使用它。
