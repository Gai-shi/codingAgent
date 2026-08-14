# Compress Tool Pressure E2E

这个 eval 用真实 LLM 跑 `ai_job`，比较启用/禁用 `compress_tool` 时的表现。

## 评测目标

这个目录现在是一个小型自然任务套件，而不是单个提示词压力测试。核心目标是测试：

1. 用户 prompt 不点名 `compress_tool`，让模型自行决定是否使用可用工具。
2. enabled / disabled 使用完全相同的用户 prompt。
3. enabled 只多暴露 `compress_tool`；disabled 通过 `--disable-compress-tool` 隐藏该工具。
4. 主要指标是正确率提升：enabled 通过、disabled 失败、且 enabled 有有效压缩。
5. 辅助指标是工具调用、上下文节省、耗时和分数差异。

runner 会临时设置较大的 `AI_JOB_CONTEXT_WINDOW`，避免自动上下文压缩掩盖 `compress_tool` 的影响。runner 不注入额外 system prompt；enabled / disabled 的行为差异只来自 `compress_tool` 是否可用。

## Case

当前套件有两个 case：

- `conflict_contract_delay`：先读取长证据材料，材料里有多版冲突契约。下一轮禁止重新读取 evidence，要求实现 `auditor.report.build_report()`。隐藏 grader 检查最终契约、region、tag、gate、owner chain、blocking condition 和 release flag 的组合是否来自有效 release/hotfix 记录。
- `trace_debug_delay`：先读取长生产 trace 和状态机材料。下一轮禁止重新读取 evidence，要求修复 `reconciler.decision.build_reconciliation_plan()`。隐藏 grader 检查 manual review、quarantine、audit defer 等多个有效生产规则。

两个 case 都采用延迟记忆结构：长工具输出出现在第一轮，最终实现发生在第二轮。

## 压力档位

使用 `--pressure` 选择压力：

```text
smoke   小噪声，验证 case 本身可完成
medium  中噪声，观察上下文/耗时差异
hard    高噪声，目标是 enabled 正确率高于 disabled
all     依次运行 smoke、medium、hard
```

旧参数 `--noise-blocks` 保留为兼容入口。传入后会覆盖当前 pressure 的默认噪声规模，主要用于临时调参。当前默认规模是 `smoke=12`、`medium=260`、`hard=760`。

## 运行

从仓库根目录运行 hard 档：

```bash
python3 evals/compress_tool_pressure_e2e/run_ai_job_ab.py \
  --output /tmp/ai_job_compress_tool_pressure \
  --force \
  --pressure hard
```

运行完整三档：

```bash
python3 evals/compress_tool_pressure_e2e/run_ai_job_ab.py \
  --output /tmp/ai_job_compress_tool_pressure \
  --force \
  --pressure all
```

只跑单个 case 的 smoke：

```bash
python3 evals/compress_tool_pressure_e2e/run_ai_job_ab.py \
  --output /tmp/ai_job_compress_tool_pressure_smoke \
  --force \
  --case-id conflict_contract_delay \
  --pressure smoke \
  --no-progress
```

使用旧噪声覆盖入口：

```bash
python3 evals/compress_tool_pressure_e2e/run_ai_job_ab.py \
  --output /tmp/ai_job_compress_tool_pressure_custom \
  --force \
  --case-id trace_debug_delay \
  --pressure hard \
  --noise-blocks 800
```

## 结果解释

输出文件：

```text
result_compress_tool_ab.json
<case_id>/<pressure>/enabled/result_enabled.json
<case_id>/<pressure>/disabled/result_disabled.json
<case_id>/<pressure>/enabled/target_repo/
<case_id>/<pressure>/disabled/target_repo/
```

核心字段：

- `summary.enabled_success_rate`: enabled 在所有 cell 中的通过率。
- `summary.disabled_success_rate`: disabled 在所有 cell 中的通过率。
- `summary.correctness_delta`: enabled 通过数减 disabled 通过数。
- `summary.compress_tool_helped_count`: enabled 通过、disabled 失败、且 enabled 有有效压缩的 cell 数量。
- `summary.enabled_compress_tool_effective_count`: enabled 成功压缩并产生上下文节省的 cell 数量。
- `summary.enabled_estimated_tool_context_chars_saved`: 粗略估算的工具输出字符节省量。
- `<case>.<pressure>.comparison.verdict`: 单个 cell 的可读结论，例如 `compress_tool_helped`、`inconclusive_both_passed`、`compress_tool_regressed`。
- `<case>.<pressure>.comparison.both_passed`: 两边都通过，说明该档压力还没拉开正确率差距。

如果 enabled 没有调用 `compress_tool`，或调用了但 `compress_tool_success_count` 为 0，本次结果不能证明工具没价值，只能说明模型没有自然成功使用它。
