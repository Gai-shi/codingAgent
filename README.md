# codingAgent

一个学习型 coding agent 原型。

## 第一版：终端聊天

已实现最小聊天闭环：

- 从项目级 `.env` 和环境变量读取认证与模型配置；
- 非流式调用模型；
- 对话历史只保存在当前进程内存里；
- 使用一个 OpenAI-compatible Chat Completions 风格接口。

## 第二版：原生 tool calling + 只读工具

当前 CLI 已开始从 chat bot 向 coding agent 过渡：

- 每次请求都会把 `read_file` 和 `grep` 作为 Chat Completions 原生 tool 传给模型；
- 如果模型返回 `tool_calls`，CLI 会在本地执行对应工具；
- 工具结果用 `role=tool`、`tool_call_id=...`、`content=<普通字符串>` 回填到 `messages`；
- CLI 会继续调用模型，直到模型返回普通 assistant 文本；
- 当前工具只支持读取或检索当前工作区内的 UTF-8 文本文件；
- 默认拒绝读取或检索 `.env`、`.git/`、`.venv/`、`.ai_job/`、`__pycache__/` 等受保护路径。
- 请求 LLM 和后台执行工具期间会关闭终端输入回显，并丢弃这段时间内误输入的内容；仅在需要用户确认时临时恢复输入回显。
- 每轮 agent loop 都会写入 Debug Trace 日志；`DebugMode` 默认按 `true` 处理，会同步把 trace 打印到终端。
- 工具调用已拆出内部契约：`ToolCall`、`ToolResult`、`BaseTool`、`ToolRegistry`、`ToolExecutor`；
- 工具调用格式已拆出 `BaseToolCallAdapter` 契约，OpenAI-compatible Chat Completions 的格式转换收口在 `OpenAIToolCallAdapter`；
- `ToolRegistry` 通过 `tools()` 暴露内部工具列表，OpenAI-compatible tool schema 由 adapter 渲染，工具层不再提供 `to_openai_schema()`；
- 工具实现已独立到 `ai_job/tools/` 包中，`chat_cli.py` 只负责 CLI 主循环和 agent turn 编排。
- 终端输入回显控制已独立到 `ai_job/terminal_input/` 包中，CLI 显式使用 `AllowInputEcho` 和 `SuppressInputEchoAndDiscard`。

当前工具定义：

```text
read_file(path: string) -> string
grep(pattern: string, path?: string, type?: string, include_protected?: boolean) -> string
```

`grep` 使用 Python 正则表达式搜索文本文件：

- `pattern`：必填，正则表达式；
- `path`：可选，限定工作区内的子目录，默认工作区根目录；
- `type`：可选，按文件扩展名过滤，例如 `py`、`kt`、`md`；默认空字符串，表示搜索所有 UTF-8 文本文件；
- `include_protected`：可选，默认 `false`。为 `true` 时表示请求检索隐藏目录或保护目录，CLI 会在本次工具执行前询问用户是否同意；
- 输出最多返回 50 条匹配，每条格式为 `relative_path:line_number:line_text`。

### Debug Trace

Trace 默认写入：

```text
.ai_job/trace.log
```

当前只记录两类最小事件：

```text
round=<轮次>
round=<轮次> tool=<工具名>
```

无论 DebugMode 取值如何，trace 都会写入日志文件。`DebugMode` 未设置时默认等价于 `true`，
会同时打印到终端。如果希望只写日志、不打印到终端：

```bash
export DebugMode=false
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

启动时会自动读取项目根目录的 `.env` 文件；真实 shell 环境变量优先级更高，不会被 `.env` 覆盖。

`.env` 示例：

```bash
OPENAI_API_KEY="你的 API Key"
OPENAI_MODEL="你的模型名"
OPENAI_BASE_URL="https://api.openai.com/v1"
AI_JOB_TIMEOUT_SECONDS="60"
AI_JOB_MAX_TOOL_ROUNDS="8"
AI_JOB_SYSTEM_PROMPT="You are a helpful coding agent. Use tools when you need workspace information."
DebugMode="false"
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

旧入口仍可使用：

```bash
python3 -m ai_job.chat_cli
```

输入 `/context` 可以查看当前进程内存里的 `messages`。

输入 `exit` / `quit` / `et` / `Ctrl-D` 退出。
