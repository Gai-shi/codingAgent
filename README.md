# codingAgent

一个学习型 coding agent 原型。

## 第一版：终端聊天

当前只实现最小聊天闭环：

- 从环境变量读取认证与模型配置；
- 非流式调用模型；
- 对话历史只保存在当前进程内存里；
- 使用一个 OpenAI-compatible Chat Completions 风格接口。

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

### 启动

```bash
python3 -m ai_job.chat_cli
```

输入 `/context` 可以查看当前进程内存里的 `messages`。

输入 `exit` / `quit` / `Ctrl-D` 退出。
