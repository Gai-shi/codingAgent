# LangGraph Runner 对照实现设计

## 背景

本项目是学习型 coding agent。现有主线已经手写实现了 agent loop、工具调用、上下文压缩、日志与会话记录。

引入 LangGraph 的目标不是替换手写实现，而是补足“主流 Agent 开发框架的实际搭建经验”，并形成一条能在面试中讲清楚的对照线：

- 手写 `AgentRunner` 展示底层原理理解。
- `LangGraphAgentRunner` 展示如何把同一套 agent loop 映射到主流框架。
- checkpoint 只做轻量验证，展示 LangGraph 的状态持久化能力，不接管项目已有会话系统。

## 范围

本次设计采用 `A -> D-light` 路线：

1. 先新增并行的 `LangGraphAgentRunner`，复刻当前 `AgentRunner` 的核心行为。
2. 再补一个最小 checkpoint 示例或可选能力点，用于展示 LangGraph 的持久化接口。

不做以下事项：

- 不移除或弱化现有手写 `AgentRunner`。
- 不让 LangGraph 接管 CLI 生命周期、日志系统或 `SessionRecorder`。
- 不把 checkpoint 变成主会话持久化方案。
- 不引入真实模型网络调用测试到默认单测。

## 架构边界

新增 `LangGraphAgentRunner`，实现现有 `BaseAgentRunner` 契约：

```python
run_turn(message_state: MessageState) -> str
```

对外仍使用项目自己的领域对象：

- `MessageState`
- `SystemMessage`
- `UserMessage`
- `SummaryMessage`
- `AssistantMessage`
- `ToolMessage`
- `ToolRegistry`
- `ToolExecutor`
- `CompressionManager`
- `SessionRecorder`

LangGraph 只负责编排单个用户回合内部的临时执行状态。`MessageState` 仍是项目长期状态的唯一来源。

## LangGraph 图结构

第一版图结构保持最小：

- `model` 节点：调用 LangChain chat model，返回 assistant 消息。
- `tools` 节点：执行 assistant 消息里的 tool calls，返回 tool 消息。
- 条件边：
  - assistant 有 tool calls：进入 `tools`
  - assistant 无 tool calls：结束本回合
- 轮数状态：
  - 记录当前工具轮数
  - 超过 `max_tool_rounds` 时抛出与现有 runner 一致的错误

## 消息转换

进入 LangGraph 前，从 `message_state.model_visible_history()` 获取模型可见历史，再转换为 LangChain 消息：

- 内部 `SystemMessage` -> LangChain `SystemMessage`
- 内部 `UserMessage` -> LangChain `HumanMessage`
- 内部 `SummaryMessage` -> LangChain `HumanMessage`
- 内部 `AssistantMessage` -> LangChain `AIMessage`
- 内部 `ToolMessage` -> LangChain `ToolMessage`

LangGraph 产生的新消息在每轮结束后转换回内部消息类型，并追加到 `message_state.history`。

## 工具执行

虽然项目已有 `tool_bridge.wrap_registry_tools()` 可以把内部工具包装成 LangChain `StructuredTool`，第一版 runner 不直接使用 LangGraph 预置 `ToolNode` 执行工具。

原因是项目里的 `compress_tool` 需要 `ToolExecutionContext(message_state=message_state)`。如果直接交给 `ToolNode`，工具节点难以自然获得项目自己的上下文对象。

因此第一版使用自定义 `tools` 节点：

1. 读取 LangChain `AIMessage.tool_calls`。
2. 转换为内部 `ToolCall`。
3. 调用现有 `ToolExecutor.execute(...)`。
4. 传入 `ToolExecutionContext(message_state=message_state)`。
5. 把结果转换为 LangChain `ToolMessage`，同时追加内部 `ToolMessage`。

这样可以保持手写 runner 和 LangGraph runner 的工具行为一致。

## CLI 入口

增加显式 runner 选择：

```text
--runner native
--runner langgraph
```

默认值为 `native`。

当用户选择 `--runner langgraph` 但未安装可选依赖时，启动阶段报清楚错误：

```text
请先安装 LangGraph 可选依赖：python3 -m pip install -e '.[langgraph]'
```

默认 `native` 路径不 import LangGraph，保持项目默认运行时无第三方依赖。

## Checkpoint D-light

checkpoint 第一版只作为轻量能力点，不进入默认 CLI 主流程。

可接受形式：

- 一个独立示例，展示如何用 LangGraph checkpointer 编译图。
- 或一个显式可选参数，仅在 `--runner langgraph` 下开启内存 checkpoint。

第一版不把 checkpoint 写入磁盘，也不替代 `SessionRecorder`。

## 测试范围

新增或调整测试时，保持范围克制：

- `tool_bridge` 单测：验证内部工具可包装为 LangChain tool。
- `LangGraphAgentRunner` 单测：
  - 无工具调用时直接返回文本。
  - 一次工具调用后返回最终文本。
  - 超过 `max_tool_rounds` 时抛错。
  - `compress_tool` 仍能拿到 `MessageState`。
- CLI factory 测试：验证默认 runner 是 `native`，显式选择 `langgraph` 时才尝试装配 LangGraph runner。

不新增依赖真实 API key 或网络请求的默认测试。

## 面试叙事

这个分支最终应能支持如下表述：

> 我先从第一性原理手写了一个 provider-agnostic coding agent，包括 agent loop、工具调用、上下文压缩和会话记录。随后我引入 LangGraph 作为并行 runner，把相同的模型-工具循环映射为 StateGraph，并比较了手写 loop 与框架 loop 在状态管理、工具执行和可扩展性上的差异。最后我用轻量 checkpoint 验证了 LangGraph 在状态持久化上的实际价值，但没有让框架覆盖项目已有的学习型架构。
