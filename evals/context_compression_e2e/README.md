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
4. 中途更新一次决策，覆盖早期 JSON 配置方案；
5. 最后一轮要求实现 `DiffReviewTool`，但不再重复关键架构类名和 marker；
5. `grader.py` 检查最终代码是否遵守早期约束和最新决策。

关键对照是：

```text
无压缩：早期约束被长噪声挤出上下文，最终容易漏掉 marker 或走向 legacy_registry / JSON / dict 注册。
有压缩：早期约束进入 summary，最终仍能保留 marker 并实现正确架构。
```

## 运行 ai_job 当前版本

从 ai_job 仓库根目录运行：

```bash
python3 evals/context_compression_e2e/run_ai_job.py \
  --output /tmp/ai_job_ctx_e2e_ai_job \
  --force
```

如需增加压力：

```bash
python3 evals/context_compression_e2e/run_ai_job.py \
  --output /tmp/ai_job_ctx_e2e_ai_job \
  --force \
  --noise-rounds 10 \
  --noise-blocks-per-round 180
```

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
  --model gpt-5.5
```

pi runner 会在长会话中插入 `/bench-compact` 命令。这个命令由
`pi_bench_compact.ts` 扩展提供，作用是触发 pi 的 compaction 并等待完成。

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

- 必须存在 `class DiffReviewTool(SentinelToolBase)`；
- 必须返回 `GuardedToolOutcome`；
- 必须通过 `CommandVault.install(...)` 注册；
- 必须使用 `MarchConfig`，不能新增 JSON 配置；
- 禁止引用 `legacy_registry`；
- 目标仓库的 `unittest` 必须通过。

## 结果解释

理想结果：

```text
ai_job 当前版本：FAIL
pi：PASS
ai_job 加入上下文压缩后：PASS
```

如果 `ai_job 当前版本` 也 PASS，优先说明模型上下文窗口仍覆盖了早期 marker；可以增加
`--noise-rounds` 或 `--noise-blocks-per-round`，直到早期约束被挤出无压缩上下文。

如果 `pi` 也 FAIL，优先看：

1. `/tmp/.../pi_turn_*_stderr.txt`；
2. 是否模型/认证不可用；
3. `/bench-compact` 是否成功；
4. 目标仓库里的 `grader.py` 输出。
