# codingAgent

一个学习型 coding agent 原型。

## 第一版：终端聊天

已实现最小聊天闭环：

- 从项目级 `.env` 和环境变量读取运行配置；
- 非流式调用模型；
- 对话历史只保存在当前进程内存里；
- 使用一个 OpenAI-compatible Chat Completions 风格接口。

## 第二版：原生 tool calling + 文件工具

当前 CLI 已开始从 chat bot 向 coding agent 过渡：

- 每次请求都会把 `read_file`、`grep` 和 `apply_patch` 作为 Chat Completions 原生 tool 传给模型；
- 如果模型返回 `tool_calls`，CLI 会在本地执行对应工具；
- 工具结果用 `role=tool`、`tool_call_id=...`、`content=<普通字符串>` 回填到 `messages`；
- CLI 会继续调用模型，直到模型返回普通 assistant 文本；
- 当前工具支持读取、检索或通过 git diff patch 修改当前工作区内的 UTF-8 文本文件；
- 默认拒绝读取、检索或修改 `.env`、`.git/`、`.venv/`、`.ai_job/`、`__pycache__/` 等受保护路径。
- 请求 LLM 和后台执行工具期间会关闭终端输入回显，并丢弃这段时间内误输入的内容；仅在需要用户确认时临时恢复输入回显。
- 每轮 agent loop 都会写入 Debug Trace 日志；`FILTER_TERMINAL_LOG_LEVEL` 默认按 `debug` 处理，会同步把全部等级日志打印到终端。
- 工具调用已拆出内部契约：`ToolCall`、`BaseTool`、`ToolRegistry`、`ToolExecutor`；工具执行结果统一为字符串，失败时返回 `Error: ...`；
- 工具调用格式已拆出 `BaseToolCallAdapter` 契约，OpenAI-compatible tool schema 和 tool_call 转换收口在 `ai_job/tool_adapters/OpenAIToolCallAdapter`；
- 内部消息格式已拆出到 `ai_job/communication/`：`SystemMessage`、`UserMessage`、`AssistantMessage`、`ToolMessage`、`MessageHistory`；
- 模型 provider 链路已拆出到 `ai_job/provider_adapters/`：`BaseChatModel.complete(...)` 返回内部 `AssistantMessage`，`OpenAIModel` 负责 OpenAI-compatible HTTP 请求和响应解析，并默认内置 `OpenAIToolCallAdapter`；
- agent loop 已拆出到 `ai_job/agent/AgentRunner`，`chat_cli.py` 只负责 CLI 主循环、对象组装和终端输入输出。
- 终端输入回显控制已独立到 `ai_job/terminal_input/` 包中，CLI 显式使用 `AllowInputEcho` 和 `SuppressInputEchoAndDiscard`。
- 日志基础设施已独立到 `ai_job/infra/logging/` 包中，通用入口为 `LogWrapper.debug/info/warn/error(TAG, text)`。
- 环境读取基础设施已独立到 `ai_job/infra/env/` 包中，`EnvLoader` 负责读取 `.env` 和 shell 环境变量，并返回扁平的 `AppEnv`。
- HTTP 请求基础设施已独立到 `ai_job/infra/http/` 包中，`OpenAIModel` 默认使用 `UrlLibHttpClient`，测试时可注入 fake `BaseHttpClient`。

当前工具定义：

```text
read_file(path: string) -> string
grep(pattern: string, path?: string, type?: string, include_protected?: boolean) -> string
apply_patch(patch: string) -> string
```

`grep` 使用 Python 正则表达式搜索文本文件：

- `pattern`：必填，正则表达式；
- `path`：可选，限定工作区内的子目录，默认工作区根目录；
- `type`：可选，按文件扩展名过滤，例如 `py`、`kt`、`md`；默认空字符串，表示搜索所有 UTF-8 文本文件；
- `include_protected`：可选，默认 `false`。为 `true` 时表示请求检索隐藏目录或保护目录，CLI 会在本次工具执行前询问用户是否同意；
- 输出最多返回 50 条匹配，每条格式为 `relative_path:line_number:line_text`。

`apply_patch` 应用 git diff 子集：

- `patch`：必填，包含 `diff --git` 文件头和 unified hunk 的 git diff 文本；
- 支持多文件修改、新增文件和删除文件；
- 不支持 rename / copy / binary patch / quoted path；
- 新增文件目标已存在时会报错，不允许覆盖；
- hunk 定位学习 pi 的做法：用旧内容唯一匹配，不依赖模型数准行号；多处匹配会失败并提示候选行号；
- 所有文件都会先完成路径校验、内容读取和内存 apply 预检查；任意预检查失败时不会写入任何文件。

### 工作区与 Debug Trace

`workspace` 是 agent 读、搜、改代码的边界：

- 默认使用启动命令时的当前目录；
- 也可以用 `--workspace` 显式指定；
- `read_file`、`grep`、`apply_patch` 都只能操作 workspace 内的文件。

例如让 agent 修另一个项目：

```bash
cd /path/to/target_project
PYTHONPATH=/Users/bytedance/Documents/AI_Projects/ai_job python3 -m ai_job
```

或：

```bash
PYTHONPATH=/Users/bytedance/Documents/AI_Projects/ai_job python3 -m ai_job --workspace /path/to/target_project
```

Trace 默认写入：

```text
<ai_job 源码根目录>/.ai_job/trace.log
```

这样即使 `--workspace` 指向不同目标项目，agent 运行日志仍会收口到 ai_job 项目根目录。

当前通过 `LogWrapper.debug("trace", text)` 记录两类最小事件：

```text
2026-08-03T20:10:00 DEBUG [trace] round=<轮次>
2026-08-03T20:10:01 DEBUG [trace] round=<轮次> tool=<工具名>
```

无论 `FILTER_TERMINAL_LOG_LEVEL` 取值如何，trace 都会写入日志文件。
`FILTER_TERMINAL_LOG_LEVEL` 未设置时默认等价于 `debug`，会同时打印全部等级日志到终端。
终端只输出等级大于等于当前过滤等级的日志，等级顺序为：

```text
debug < info < warn < error < none
```

对应行为：

```text
debug: debug / info / warn / error
info : info / warn / error
warn : warn / error
error: error
none : 不输出任何日志到终端
```

如果希望只写日志、不打印到终端：

```bash
export FILTER_TERMINAL_LOG_LEVEL=none
```

### 当前进度

已验证当前 CLI 可以通过本机 ModelHub 代理完成一次多轮聊天：

- `OPENAI_BASE_URL=http://127.0.0.1:8787/v1`
- `OPENAI_MODEL=gpt-5.5`
- 当前代码实际请求 `http://127.0.0.1:8787/v1/chat/completions`

这里的 ModelHub 接入方式不是在项目内实现独立的 ModelHub provider，而是复用本机代理提供的
OpenAI-compatible Chat Completions 接口。项目内部现在只依赖 Chat Completions 的：

- `messages`
- `tools`
- `assistant.tool_calls`
- `role=tool` 工具结果消息

### 环境变量

启动时会自动读取 ai_job 源码根目录的 `.env` 文件；真实 shell 环境变量优先级更高，不会被 `.env` 覆盖。
环境变量读取已集中在 `ai_job/infra/env/env_loader.py` 中，当前返回的运行配置对象为 `AppEnv`。

`.env` 示例：

```bash
OPENAI_API_KEY="你的 API Key"
OPENAI_MODEL="你的模型名"
OPENAI_BASE_URL="https://api.openai.com/v1"
AI_JOB_TIMEOUT_SECONDS="60"
AI_JOB_MAX_TOOL_ROUNDS="8"
AI_JOB_SYSTEM_PROMPT="You are a helpful coding agent. Use tools when you need workspace information."
FILTER_TERMINAL_LOG_LEVEL="debug"
```

如果使用本机 ModelHub 代理，可以把 `.env` 改成：

```bash
OPENAI_API_KEY="任意非空占位值"
OPENAI_BASE_URL="http://127.0.0.1:8787/v1"
OPENAI_MODEL="gpt-5.5"
```

当前 `.env` loader 支持空行、`#` 注释行、`KEY=VALUE`、`export KEY=VALUE`，以及单引号/双引号包裹的值。`.env` 已被 `.gitignore` 忽略；不要把真实密钥写入代码或提交到仓库。

### 启动

推荐从包入口启动：

```bash
python3 -m ai_job
```

默认 workspace 是当前目录；如果要在任意目录启动并指定目标项目：

```bash
PYTHONPATH=/Users/bytedance/Documents/AI_Projects/ai_job python3 -m ai_job --workspace /path/to/target_project
```

旧入口仍可使用：

```bash
python3 -m ai_job.chat_cli --workspace /path/to/target_project
```

输入 `/context` 可以查看当前进程内存里的 `messages`。

输入 `exit` / `quit` / `et` / `Ctrl-D` 退出。
