# codingAgent

一个学习型 coding agent 原型。

## 第一版：终端聊天

已实现最小聊天闭环：

- 从环境变量读取认证与模型配置；
- 非流式调用模型；
- 对话历史只保存在当前进程内存里；
- 使用一个 OpenAI-compatible Chat Completions 风格接口。

## 第二版：原生 tool calling + read_file

当前 CLI 已开始从 chat bot 向 coding agent 过渡：

- 每次请求都会把 `read_file` 作为 Chat Completions 原生 tool 传给模型；
- 如果模型返回 `tool_calls`，CLI 会在本地执行对应工具；
- 工具结果用 `role=tool`、`tool_call_id=...`、`content=<普通字符串>` 回填到 `messages`；
- CLI 会继续调用模型，直到模型返回普通 assistant 文本；
- 第一版工具只支持读取当前工作区内的 UTF-8 文本文件；
- 默认拒绝读取 `.env`、`.git/`、`__pycache__/` 等受保护路径。
- 等待模型返回期间会关闭终端输入回显，并丢弃这段时间内误输入的内容，避免用户输入和模型输出重叠。
- 每轮 agent loop 都会写入 Debug Trace 日志；`DebugMode` 默认按 `true` 处理，会同步把 trace 打印到终端。

当前工具定义：

```text
read_file(path: string) -> string
```

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

必填：

```bash
export OPENAI_API_KEY="你的 API Key"
export OPENAI_MODEL="你的模型名"
```

可选：

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
export AI_JOB_TIMEOUT_SECONDS="60"
export AI_JOB_MAX_TOOL_ROUNDS="8"
export AI_JOB_SYSTEM_PROMPT="You are a helpful coding agent. Use tools when you need workspace information."
export DebugMode="false"
```

如果使用本机 ModelHub 代理，可以改成：

```bash
export OPENAI_BASE_URL="http://127.0.0.1:8787/v1"
export OPENAI_MODEL="gpt-5.5"
```

注意：`OPENAI_API_KEY` 仍然需要设置，因为当前 CLI 会在启动时校验它存在；不要把真实密钥写入代码或提交到仓库。

### 启动

```bash
python3 -m ai_job.chat_cli
```

输入 `/context` 可以查看当前进程内存里的 `messages`。

输入 `exit` / `quit` / `Ctrl-D` 退出。
