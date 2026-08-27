# LangGraph Runner 实现计划

## 文件结构

- `ai_job/agent/langgraph_runner/message_conversion.py`
  - 负责内部消息类型和 LangChain 消息类型之间的双向转换。
- `ai_job/agent/langgraph_runner/runner.py`
  - 实现 `LangGraphAgentRunner`，对外满足 `BaseAgentRunner.run_turn(...)`。
- `ai_job/agent/langgraph_runner/checkpoint_demo.py`
  - 提供最小 checkpoint 示例入口，不接管 CLI 主流程。
- `ai_job/agent/langgraph_runner/__init__.py`
  - 导出 LangGraph runner 相关公开对象。
- `ai_job/composition/cli_factory.py`
  - 增加 runner 选择参数，并按需装配 native 或 langgraph runner。
- `ai_job/chat_cli.py`
  - 增加 `--runner` CLI 参数，并透传到 composition。
- `README.zh-CN.md`
  - 记录 LangGraph 可选依赖、启动方式和学习定位。
- `README.md`
  - 同步英文说明。
- `tests/test_langgraph_message_conversion.py`
  - 覆盖内部消息和 LangChain 消息转换。
- `tests/test_langgraph_tool_bridge.py`
  - 覆盖内部工具包装成 LangChain tool。
- `tests/test_langgraph_agent_runner.py`
  - 覆盖 LangGraph runner 的回合行为。
- `tests/test_cli_factory.py`
  - 覆盖 runner 选择和依赖隔离。
- `tests/test_chat_cli_workspace.py`
  - 覆盖 `--runner` 参数解析。

## 任务

1. 在 `tests/test_langgraph_message_conversion.py` 写转换测试，覆盖 `SystemMessage`、`UserMessage`、`SummaryMessage`、`AssistantMessage`、`ToolMessage` 到 LangChain 消息的转换，以及 LangChain `AIMessage` tool calls 到内部 `AssistantMessage` 的转换。

2. 运行 `python3 -m unittest tests.test_langgraph_message_conversion`，确认测试因缺少转换模块而失败。

3. 在 `ai_job/agent/langgraph_runner/message_conversion.py` 实现消息转换模块，确保 `SummaryMessage` 的渲染文本与现有 `OpenAIModel` 保持一致。

4. 再次运行 `python3 -m unittest tests.test_langgraph_message_conversion`，确认转换测试通过。

5. 在 `tests/test_langgraph_tool_bridge.py` 写工具桥接测试，覆盖 `wrap_tool(...)` 暴露工具名称、描述、参数 schema，并能调用内部 `BaseTool.execute(...)`。

6. 运行 `python3 -m unittest tests.test_langgraph_tool_bridge`，确认现有 `tool_bridge.py` 行为通过；若受 LangChain 版本接口影响失败，只调整桥接层，不改变内部 `BaseTool` 契约。

7. 在 `tests/test_langgraph_agent_runner.py` 写 `LangGraphAgentRunner` 测试，使用可脚本化的假 LangChain chat model，覆盖无工具调用时直接返回文本。

8. 继续在 `tests/test_langgraph_agent_runner.py` 写一次工具调用测试，验证工具结果追加为内部 `ToolMessage`，最终 assistant 文本追加回 `message_state.history`。

9. 继续在 `tests/test_langgraph_agent_runner.py` 写最大工具轮数测试，验证超过 `max_tool_rounds` 时抛出与 native runner 一致的中文错误。

10. 继续在 `tests/test_langgraph_agent_runner.py` 写 `compress_tool` 上下文测试，验证工具执行节点传入 `ToolExecutionContext(message_state=message_state)`，压缩结果写回原始 `MessageState`。

11. 运行 `python3 -m unittest tests.test_langgraph_agent_runner`，确认测试因缺少 runner 实现而失败。

12. 在 `ai_job/agent/langgraph_runner/runner.py` 实现 `LangGraphAgentRunner`，使用 LangGraph 表达 `model` 节点、`tools` 节点和条件边，并复用现有 `ToolExecutor`、`CompressionManager`、`SessionRecorder`。

13. 再次运行 `python3 -m unittest tests.test_langgraph_agent_runner`，确认 LangGraph runner 测试通过。

14. 在 `ai_job/agent/langgraph_runner/__init__.py` 导出 `LangGraphAgentRunner` 和消息转换必要对象，保持默认 `ai_job.agent` import 不强制加载 LangGraph。

15. 在 `tests/test_cli_factory.py` 增加 native 默认 runner 测试，验证不传 runner 选择时仍装配 `AgentRunner`。

16. 在 `tests/test_cli_factory.py` 增加 langgraph runner 选择测试，验证显式选择时才导入并装配 `LangGraphAgentRunner`。

17. 在 `tests/test_cli_factory.py` 增加缺少 LangGraph 依赖时的错误测试，验证错误信息包含安装命令 `python3 -m pip install -e '.[langgraph]'`。

18. 在 `ai_job/composition/cli_factory.py` 增加 runner 选择参数，默认 `native`，并将 LangGraph import 放在选择分支内部。

19. 在 CLI 参数测试文件中增加 `--runner` 解析测试，覆盖默认值和 `langgraph` 值。

20. 在 `ai_job/chat_cli.py` 增加 `--runner` 参数，并把选择传给 `create_cli_runtime(...)`。

21. 运行 CLI 和 composition 相关测试，命令为 `python3 -m unittest tests.test_cli_factory tests.test_chat_cli_workspace`，确认通过。

22. 在 `ai_job/agent/langgraph_runner/checkpoint_demo.py` 增加最小 checkpoint 示例，展示 LangGraph graph compile 时接入内存 checkpointer，不写入磁盘，不替代 `SessionRecorder`。

23. 在对应测试文件中增加 checkpoint-light 测试，验证示例可以构建带 checkpointer 的图对象；测试不依赖真实模型和网络。

24. 更新 `README.zh-CN.md`，说明 LangGraph 是可选框架对照实现，给出安装 extra 和 `--runner langgraph` 启动命令。

25. 更新 `README.md`，同步英文版 LangGraph 可选框架说明。

26. 运行完整单测 `python3 -m unittest`，预期全部通过；如果本地未安装 LangGraph optional extra，记录无法运行 LangGraph 相关测试的原因。

27. 运行 `git status --short`，确认只包含本轮计划内文件变更。
