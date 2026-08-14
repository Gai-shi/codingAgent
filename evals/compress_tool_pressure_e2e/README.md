# Compress Tool Pressure E2E

这个 eval 用真实 LLM 跑 `ai_job`，比较启用/禁用 `compress_tool` 时的表现。

## 评测目标

这个目录现在是一个小型自然任务套件，而不是单个提示词压力测试。核心目标是测试：

1. `neutral` 策略下，用户 prompt 不点名 `compress_tool`，让模型自行决定是否使用可用工具。
2. `guided` 策略下，只在 enabled 组加入轻量工具使用提示，disabled 组不注入无法使用的工具提示。
3. enabled 暴露 `compress_tool`；disabled 通过 `--disable-compress-tool` 隐藏该工具。
4. 主要指标是无损提效：enabled 正确率不低于 disabled，且两边都通过时 enabled 有有效压缩并更快。
5. 辅助指标是正确率提升、工具调用、上下文节省、耗时和分数差异。

runner 默认不覆盖 `AI_JOB_CONTEXT_WINDOW`，让 ai_job 使用正常自动上下文压缩行为。默认 `--tool-policy neutral` 保持 enabled / disabled 的 prompt 完全一致；`--tool-policy guided` 用来测试“工具可用 + 合理策略提示”后的收益。如果需要复现“自动压缩不兜底，只看 compress_tool 主动压缩”的压力口径，可以显式传 `--auto-compression-context-window 10000000`。

## Case

当前套件有两个 case：

- `conflict_contract_delay`：第一轮只从 `evidence/00_index.txt` 进入默认 release 交接包，并刻意不跟进 post-lock route。第二轮用户指出默认包是 pre-lock 资料，要求按 route 读取 errata，再按消歧 route 先检查候选/回滚长文件，再定位 legacy/draft archive 和 `final_contract_delta.txt`。hard 档第三轮会先读取 `README.md`、`auditor/report.py`、`auditor/schema.py`、`auditor/formatting.py` 和 tests，整理最终约束但不改代码。最后一轮禁止重新读取 evidence，要求实现 Python 3.9 兼容的 `auditor.report.build_report()`，并保持 report/schema/formatting 的职责边界。隐藏 grader 检查最终契约、region、tag、gate、owner chain、blocking condition、release flag、handoff ticket、review signoff、rollback guard 和 control tag 的组合是否来自有效 release/hotfix 记录，并拒绝默认包/候选包值。
- `trace_debug_delay`：第一轮只从 `evidence/00_index.txt` 进入默认 incident triage 包，并刻意不跟进 post-lock route。第二轮用户指出默认包是 pre-lock 资料，要求按 route 读取 errata，再按消歧 route 先检查 replay/candidate 长文件，再定位 production archive、state rules 和 `active_trace_manifest.txt`。hard 档第三轮会先读取 `README.md`、`reconciler/decision.py`、`reconciler/rules.py` 和 tests，整理最终约束但不改代码。最后一轮禁止重新读取 evidence，要求修复 Python 3.9 兼容的 `reconciler.decision.build_reconciliation_plan()`，并保持 decision/rules 的职责边界。隐藏 grader 检查 manual review、quarantine、audit defer、rule id、resolver group、audit tag、suppression 和默认 non-active 计划等多个有效生产规则，并拒绝默认 triage/候选规则值。

两个 case 都采用延迟记忆结构：误导长输出出现在第一轮，纠偏长输出出现在第二轮，hard 档增加实现前约束整理轮，最终实现发生在最后一轮。smoke/medium 保持三轮，用于快速回归和调参。

## 压力档位

使用 `--pressure` 选择压力：

```text
smoke   小噪声，三轮流程，验证 case 本身可完成
medium  中噪声，三轮流程，观察上下文/耗时差异
hard    高噪声、四轮深项目流程，目标是 enabled 正确率不下降并更快
all     依次运行 smoke、medium、hard
```

旧参数 `--noise-blocks` 保留为兼容入口。传入后会覆盖当前 pressure 的默认噪声规模，主要用于临时调参。当前默认规模是 `smoke=12`、`medium=260`、`hard=1600`。

## 工具策略

使用 `--tool-policy` 选择工具提示策略：

```text
neutral  默认值。enabled / disabled 使用完全相同的自然任务 prompt，用于测纯工具可用性。
guided   只给 enabled 组加入轻量 compress_tool 使用提示，用于测工具产品化引导后的收益。
all      依次运行 neutral 和 guided，并在 summary 中同时给出 pure_availability_delta 与 guided_tool_delta。
```

## 运行

从仓库根目录运行 hard 档：

```bash
python3 evals/compress_tool_pressure_e2e/run_ai_job_ab.py \
  --output /tmp/ai_job_compress_tool_pressure \
  --force \
  --pressure hard
```

runner 默认并行执行当前策略下的 enabled/disabled cell，`--max-workers` 默认是 4。终端默认只输出摘要、结果 JSON 路径和运行日志路径；完整 JSON 写入 `result_compress_tool_ab.json`，过程日志写入 `run_ai_job_ab.log`。需要实时终端进度时可显式加 `--progress`。

并行运行时，runner 会为每个 variant 注入独立的 ai_job session/log 基准路径，避免多个 ai_job 进程在同一毫秒启动时把 `.ai_job/sessions` 或 `.ai_job/logs` 写到同一个文件，导致 enabled/disabled 诊断串台。

运行带工具策略提示的 hard 档：

```bash
python3 evals/compress_tool_pressure_e2e/run_ai_job_ab.py \
  --output /tmp/ai_job_compress_tool_pressure_guided \
  --force \
  --pressure hard \
  --tool-policy guided
```

运行同一策略但显式绕开自动上下文压缩兜底：

```bash
python3 evals/compress_tool_pressure_e2e/run_ai_job_ab.py \
  --output /tmp/ai_job_compress_tool_pressure_guided_no_auto \
  --force \
  --pressure hard \
  --tool-policy guided \
  --auto-compression-context-window 10000000
```

同时跑自然选择和带引导两组：

```bash
python3 evals/compress_tool_pressure_e2e/run_ai_job_ab.py \
  --output /tmp/ai_job_compress_tool_pressure_policy_ab \
  --force \
  --pressure hard \
  --tool-policy all
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
run_ai_job_ab.log
policy_results.<policy>.summary
policy_results.<policy>.cases
<case_id>/<pressure>/enabled/result_enabled.json
<case_id>/<pressure>/disabled/result_disabled.json
<case_id>/<pressure>/enabled/target_repo/
<case_id>/<pressure>/disabled/target_repo/
```

当 `--tool-policy all` 时，每个策略会写入独立子目录：

```text
neutral/<case_id>/<pressure>/enabled/result_enabled.json
guided/<case_id>/<pressure>/enabled/result_enabled.json
```

核心字段：

- `summary.enabled_success_rate`: enabled 在所有 cell 中的通过率。
- `summary.disabled_success_rate`: disabled 在所有 cell 中的通过率。
- `summary.correctness_delta`: enabled 通过数减 disabled 通过数。
- `summary.pure_availability_delta`: `neutral` 策略下的正确率差异；只在 `--tool-policy all` 汇总中出现。
- `summary.guided_tool_delta`: `guided` 策略下的正确率差异；只在 `--tool-policy all` 汇总中出现。
- `summary.compress_tool_helped_count`: enabled 通过、disabled 失败、且 enabled 有有效压缩的 cell 数量。
- `summary.lossless_speedup_count`: enabled 和 disabled 都通过、enabled 有有效压缩、且 enabled 耗时更短的 cell 数量。
- `summary.latency_ratio`: enabled 总耗时 / disabled 总耗时；小于 1 表示 enabled 更快。
- `summary.enabled_compress_tool_effective_count`: enabled 成功压缩并产生上下文节省的 cell 数量。
- `summary.enabled_estimated_tool_context_chars_saved`: 粗略估算的工具输出字符节省量。
- `<case>.<pressure>.comparison.verdict`: 单个 cell 的可读结论，例如 `lossless_speedup`、`compress_tool_helped`、`inconclusive_both_passed`、`compress_tool_regressed`。
- `<case>.<pressure>.comparison.both_passed`: 两边都通过，说明该档压力还没拉开正确率差距。

如果 enabled 没有调用 `compress_tool`，或调用了但 `compress_tool_success_count` 为 0，本次结果不能证明工具没价值，只能说明模型没有自然成功使用它。
