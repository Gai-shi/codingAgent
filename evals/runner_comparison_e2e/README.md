# Runner Comparison E2E

这个 eval 用真实 CLI 对比 `native` 和 `langgraph` 两个 runner。

## 评测目标

一次命令喂同一批任务，并分别运行：

- `python -m ai_job --runner native`
- `python -m ai_job --runner langgraph`

最终自动汇总三类差异：

- 正确率：每个 case 是否通过确定性 grader。
- Token：使用项目现有 `字符数 / 4` 估算器，不是 API 账单 usage。
- 用时：子进程端到端 wall-clock 时间，包括 CLI 启动、模型请求、工具执行和退出。

这个 eval 是端到端真实对比。`native` 使用项目自己的 `OpenAIModel`，
`langgraph` 使用 `langchain_openai.ChatOpenAI`，因此结果不代表纯框架 overhead。

## 安装

LangGraph runner 需要可选依赖：

```bash
.venv/bin/python -m pip install -e '.[langgraph]'
```

如果没有安装这组依赖，`langgraph` variant 会在结果 JSON 中表现为
`exit_code=2`，并在 `diagnostics.langgraph_dependency_error_count` 中记录启动失败。

同时需要正常配置模型环境变量：

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

## 运行

完整运行 6 个 case：

```bash
.venv/bin/python -m evals.runner_comparison_e2e.run_ai_job_ab \
  --output /tmp/runner_eval \
  --force
```

只跑单个 case：

```bash
.venv/bin/python -m evals.runner_comparison_e2e.run_ai_job_ab \
  --output /tmp/runner_eval \
  --force \
  --case-id single_file_patch
```

默认串行运行，方便比较耗时。如果你更关心更快跑完，可以加：

```bash
--max-workers 2
```

如果想看实时进度：

```bash
--progress
```

## Case

- `direct_answer`：不需要工具，直接回答固定验证码。
- `read_file_answer`：读取指定文件并回答隐藏值。
- `grep_then_read`：先搜索 sentinel，再读取命中文件并回答。
- `single_file_patch`：修复一个 Python 函数。
- `multi_file_patch`：跨两个 Python 文件补齐行为。
- `tool_error_recovery`：从过期路径失败中恢复，搜索真实文件并回答。

## 输出

主要看：

```bash
cat /tmp/runner_eval/result_runner_comparison_ab.json
```

过程日志：

```bash
cat /tmp/runner_eval/run_ai_job_ab.log
```

某个 case 的单 runner 结果：

```bash
cat /tmp/runner_eval/single_file_patch/native/result_native.json
cat /tmp/runner_eval/single_file_patch/langgraph/result_langgraph.json
```

结果目录结构：

```text
<output>/
  result_runner_comparison_ab.json
  run_ai_job_ab.log
  <case_id>/
    prompts/
    prompt_manifest.json
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

## 关键字段

- `native_success_rate` / `langgraph_success_rate`
- `langgraph_correctness_delta`
- `native_estimated_total_tokens` / `langgraph_estimated_total_tokens`
- `estimated_total_tokens_delta`
- `langgraph_to_native_token_ratio`
- `native_elapsed_seconds` / `langgraph_elapsed_seconds`
- `elapsed_seconds_delta`
- `langgraph_to_native_latency_ratio`
