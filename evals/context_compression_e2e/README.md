# Context Compression E2E Benchmark

这个目录提供一个真实 E2E 评测，用来验证：

```text
当前 ai_job 无上下文压缩：预期失败
pi 显式压缩上下文：预期成功
修复后的 ai_job：预期成功
```

它不是 mock benchmark，会真实调用被测 coding agent 和真实 LLM。

## 测试思想

同一个任务被拆成多轮长会话：

1. 第一轮给出真正有效的架构约束；
2. 早期约束里包含一个最终任务不会重复的 `CONTEXT_RETENTION_MARKER`；
3. 中间插入大量“废弃设计/旧日志”噪声和伪造旧 marker；
4. 中途追加实现拓扑决策：`DiffReviewTool` 必须进入 `sentinel_lab/audit/` 多文件布局，而不是旧扁平路径；
5. 中途追加 parser contract：`UnifiedDiffSummary` frozen dataclass、parser marker、禁止 dict parser；
6. 中途更新一次配置决策，覆盖早期 JSON 配置方案，并给出最终任务不会重复的 `CONFIG_RETENTION_MARKER`；
7. 后段追加 warning taxonomy：精确 warning/error code、payload 形状、policy version；
8. 后段追加 audit metadata：audit channel、payload schema version、warning source、metadata marker；
9. 最后一轮要求实现 `DiffReviewTool`，但不再重复精确 marker、warning code、metadata 常量、parser 类型名、policy version 或文件拓扑；
10. `grader.py` 检查最终代码是否遵守早期约束和所有阶段性最新决策。

关键对照是：

```text
无压缩：完整历史必须原样进入后续请求；当真实历史超过模型上下文窗口时，任务会失败。
有压缩：早期约束进入 summary，后续请求不需要携带完整噪声历史，最终仍能保留 marker 并实现正确架构。
```

目标 fixture 已从单文件小项目升级为中等规模项目：它包含 canonical `core`、
`audit` 包、`experimental` 干扰包、`future_registry`、function registry adapter
和多份冲突文档。通过测试不再只靠两个 marker，还需要保留多阶段决策：

- 扁平旧路径 vs `sentinel_lab/audit/` 拓扑；
- dict parser vs `UnifiedDiffSummary` frozen dataclass；
- JSON config vs `MarchConfig`；
- legacy/function/future registry vs `CommandVault.install`；
- 旧 warning code vs `W-MARCH-*` / `E-MARCH-*` taxonomy；
- 旧 metadata vs `MARCH-AUDIT-CHANNEL-42` / `MARCH-PAYLOAD-SCHEMA-12` / `march-warning-ledger-17`。

## 运行 ai_job 当前版本

从 ai_job 仓库根目录运行：

```bash
python3 evals/context_compression_e2e/run_ai_job.py \
  --output /tmp/ai_job_ctx_e2e_ai_job \
  --force
```

如需增加压力。下面这组约为百万 token 级累计历史，通常更适合区分“无压缩”和“定期压缩”：

```bash
python3 evals/context_compression_e2e/run_ai_job.py \
  --output /tmp/ai_job_ctx_e2e_ai_job \
  --force \
  --noise-rounds 32 \
  --noise-blocks-per-round 192 \
  --compact-every 4
```

注意：如果模型上下文窗口足够大，即使 ai_job 没有压缩能力也可能通过。这种
结果是合理的：任务能放进窗口时，本来就不必为了压缩而压缩。

## 真实超长历史压力测试

如果你要验证“这个任务不压缩就无法取得预期效果”，应让**真实发送的原始历史**
超过被测模型的上下文窗口，而不是裁剪模型可见上下文。可以用
`--min-raw-history-chars` 自动增加噪声轮数：

```bash
python3 evals/context_compression_e2e/run_ai_job.py \
  --output /tmp/ai_job_ctx_e2e_ai_job_overflow \
  --force \
  --noise-blocks-per-round 256 \
  --min-raw-history-chars 1200000 \
  --compact-every 4
```

这仍然调用真实 LLM，不做 fake，不裁剪上下文。当前无压缩 ai_job 会把完整
历史原样发送给 provider；如果超过模型真实窗口，应出现 context-length 错误
或最终无法完成任务。

对应的 pi 对照：

```bash
python3 evals/context_compression_e2e/run_pi.py \
  --output /tmp/ai_job_ctx_e2e_pi_overflow \
  --force \
  --provider openai \
  --model gpt-5.5 \
  --noise-blocks-per-round 256 \
  --min-raw-history-chars 1200000 \
  --compact-every 4
```

生成的 result JSON 会包含：

- `noise_rounds`：为达到目标原始历史长度实际生成的噪声轮数；
- `prompt_stats.raw_ai_job_user_history_chars`：无压缩 ai_job 等价原始用户历史字符数；
- `diagnostics`：从 ai_job stderr / session record 提取的高信号失败原因，例如 context length、模型权限、apply_patch 格式或 hunk 计数错误；
- `grade`：最终仓库判分。

## 运行 pi

默认假设 pi 在：

```text
/Users/bytedance/Documents/AI_Projects/storage/pi/pi-test.sh
```

运行：

```bash
python3 evals/context_compression_e2e/run_pi.py \
  --output /tmp/ai_job_ctx_e2e_pi \
  --force
```

如果需要指定模型：

```bash
python3 evals/context_compression_e2e/run_pi.py \
  --output /tmp/ai_job_ctx_e2e_pi \
  --force \
  --provider openai \
  --model gpt-5.5 \
  --noise-rounds 32 \
  --noise-blocks-per-round 192 \
  --compact-every 4
```

pi runner 会在长会话中插入 `/bench-compact` 命令。这个命令由
`pi_bench_compact.ts` 扩展提供，作用是触发 pi 的 compaction 并等待完成。`--compact-every 4` 表示每 4 个非压缩 turn 后压缩一次。

## 只生成 fixture 和 prompts

```bash
python3 -m evals.context_compression_e2e.benchmark_case \
  --output /tmp/ai_job_ctx_fixture \
  --force \
  --include-compact-turns
```

生成目录包含：

```text
target_repo/           被修改的目标仓库，不包含外部 grader，避免泄漏隐藏 marker
prompts/               多轮 prompt 文件
prompt_manifest.json   prompt 顺序
```

## 判分

对任意生成后的目标仓库执行外部 grader：

```bash
python3 evals/context_compression_e2e/grader.py \
  --target /tmp/ai_job_ctx_e2e_pi/target_repo
```

判分包括：

- 必须在 `sentinel_lab/audit/diff_review_tool.py` 存在 `class DiffReviewTool(SentinelToolBase)`；
- 必须存在 `sentinel_lab/audit/unified_diff_parser.py`、`sentinel_lab/audit/warning_policy.py` 和 `sentinel_lab/audit/audit_metadata.py`；
- `sentinel_lab/audit/__init__.py` 必须导出 `DiffReviewTool`；
- parser 必须定义 `@dataclass(frozen=True) UnifiedDiffSummary`；
- `parse_unified_diff(...)` 必须返回 `UnifiedDiffSummary` 而非 dict；
- parser 必须保留 `PARSER_RETENTION_MARKER = "MARCH-PARSER-7731"`；
- 必须返回 `GuardedToolOutcome`；
- 必须引用最新配置形态 `MarchConfig`；
- 必须声明 `name = "diff_review"`；
- `execute` 必须标注返回 `GuardedToolOutcome`；
- 必须通过 `CommandVault.install(...)` 注册；
- 必须用 `MarchConfig(audit_label="march-diff-review", policy_version="MARCH-AUDIT-V7")` 安装；
- 必须保留早期 `CONTEXT_RETENTION_MARKER`、拓扑 `TOPOLOGY_RETENTION_MARKER`、中途 `CONFIG_RETENTION_MARKER`、policy `POLICY_RETENTION_MARKER` 和 metadata `METADATA_RETENTION_MARKER`；
- 必须保留 `W-MARCH-FILE-337`、`W-MARCH-TODO-214`、`E-MARCH-STRICT-901`、`E-MARCH-EMPTY-044`；
- 必须保留 `MARCH-AUDIT-CHANNEL-42`、`MARCH-PAYLOAD-SCHEMA-12`、`march-warning-ledger-17`；
- 成功 payload 必须包含 `audit_channel`、`payload_schema_version`、`audit_label`；
- 不能新增 JSON 配置；
- 禁止引用 `legacy_registry`、`future_registry`、`FunctionRegistry` 或 `sentinel_lab.experimental`；
- 禁止回退到噪声中的 `BaseTool` / `ToolResult`；
- 目标仓库的 `unittest` 必须通过。

## 结果解释

理想结果：

```text
ai_job 当前版本：FAIL
pi：PASS
ai_job 加入上下文压缩后：PASS
```

如果 `ai_job 当前版本` 也 PASS，优先说明这次任务仍然落在模型可承载范围内；
可以增加 `--min-raw-history-chars`，直到原始历史超过被测模型真实上下文窗口。

如果 `grade` 是 FAIL，先看 `result_ai_job.json` 里的 `diagnostics`：

- `context_length_exceeded_count > 0`：说明无压缩历史已经超过真实模型窗口；
- `apply_patch_begin_patch_format_error_count > 0`：说明模型把 Codex 风格 `*** Begin Patch` 误传给 ai_job 的 git-diff-only `apply_patch`；
- `apply_patch_hunk_count_error_count > 0`：说明模型生成的 git diff hunk 行数不一致；
- `model_permission_error_count > 0`：说明本次还混入了 provider/model 权限问题，不能只按代码能力解释。

如果 `pi` 也 FAIL，优先看：

1. `/tmp/.../pi_turn_*_stderr.txt`；
2. 是否模型/认证不可用；
3. `/bench-compact` 是否成功；
4. 目标仓库里的 `grader.py` 输出。
