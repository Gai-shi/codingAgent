# codingAgent

一个学习型 coding agent 原型。

## 第一版：终端聊天

当前只实现最小聊天闭环：

- 从环境变量读取认证与模型配置；
- 非流式调用模型；
- 对话历史只保存在当前进程内存里；
- 使用一个 OpenAI-compatible Chat Completions 风格接口。

### 当前进度

已验证当前 CLI 可以通过本机 ModelHub 代理完成一次多轮聊天：

- `OPENAI_BASE_URL=http://127.0.0.1:8787/v1`
- `OPENAI_MODEL=gpt-5.5`
- 当前代码实际请求 `http://127.0.0.1:8787/v1/chat/completions`

这里的 ModelHub 接入方式不是在项目内实现独立的 ModelHub provider，而是复用本机代理提供的
OpenAI-compatible Chat Completions 接口。项目内部仍然只理解 `messages -> assistant text`
这一层最小协议。

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
export AI_JOB_SYSTEM_PROMPT="You are a helpful assistant."
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
