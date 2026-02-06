# pico-agent

Protocol-based AI agent framework with dependency injection via pico-ioc. LLM-agnostic, multi-provider.

## Commands

```bash
pip install -e .                  # Install in dev mode
pytest tests/ -v                  # Run tests
pytest --cov=pico_agent --cov-report=term-missing tests/  # Coverage
tox                               # Full matrix (3.11-3.14)
mkdocs serve -f mkdocs.yml        # Local docs
```

## Project Structure

```
src/pico_agent/
  __init__.py          # Public API exports
  config.py            # AgentConfig, AgentType, AgentCapability, LLMConfig, ToolConfig
  decorators.py        # @agent, @tool decorators with metadata keys
  interfaces.py        # Protocols: Agent, LLM, LLMFactory, CentralConfigClient
  registry.py          # ToolRegistry, LocalAgentRegistry, AgentConfigService
  scanner.py           # AgentScanner, ToolScanner (auto-discovery via @configure)
  router.py            # ModelRouter (capability → model name)
  proxy.py             # DynamicAgentProxy, TracedAgentProxy (runtime invocation)
  messages.py          # build_messages() shared message builder
  locator.py           # AgentLocator factory (creates agent proxies)
  providers.py         # LangChainAdapter, LangChainLLMFactory (multi-provider LLM)
  tools.py             # ToolWrapper, AgentAsTool (tool abstraction)
  virtual.py           # VirtualAgentRunner, VirtualAgentManager (runtime agents)
  virtual_tools.py     # DynamicTool, VirtualToolManager (runtime tools)
  interceptor.py       # AgentInterceptor (MethodInterceptor for agent calls)
  scheduler.py         # PlatformScheduler (asyncio.Semaphore concurrency)
  tracing.py           # TraceService, TraceRun (observability)
  experiments.py       # ExperimentRegistry (A/B testing)
  validation.py        # AgentValidator (config validation)
  logging.py           # get_logger, configure_logging
  exceptions.py        # AgentError, AgentDisabledError, AgentConfigurationError
```

## Key Concepts

### Agent Definition
- **`@agent(name, capability, system_prompt, ...)`**: Marks a Protocol class as an agent. Metadata stored in `AGENT_META_KEY`
- **`AgentType`**: `ONE_SHOT` (single LLM call), `REACT` (tool loop via langgraph), `WORKFLOW` (map-reduce)
- **`AgentCapability`**: `FAST`, `SMART`, `REASONING`, `VISION`, `CODING` — mapped to model names by ModelRouter

### Tool Definition
- **`@tool(name, description)`**: Marks a class as a tool. Metadata stored in `TOOL_META_KEY`
- **`ToolWrapper`**: Adapts pico tools to LangChain tool interface
- **`AgentAsTool`**: Wraps child agents as invocable tools

### Proxy & Execution
- **`DynamicAgentProxy`**: Creates runtime callables matching protocol method signatures
- **`TracedAgentProxy`**: Adds tracing to proxy execution
- **Runtime model override**: `_model="gpt-4"` kwarg on any agent method call
- **Structured output**: If return type is a Pydantic model, response is parsed automatically

### LLM Providers
- **`LangChainLLMFactory`**: Multi-provider factory (OpenAI, Azure, Google, Anthropic, DeepSeek, Qwen)
- **Provider detection**: Auto-detects from model name prefix (e.g., `claude-` → anthropic)
- **Explicit syntax**: `"provider:model"` (e.g., `"openai:gpt-5-mini"`)
- **`LLMConfig`**: Centralized API keys and base URLs

### Configuration & Registry
- **`AgentConfigService`**: Merges central + local + runtime config (priority: central > local > runtime)
- **`LocalAgentRegistry`**: Stores agent configs discovered by scanner
- **`ToolRegistry`**: Manages tool instances, supports tag-based lookup

### Virtual Agents
- **`VirtualAgentRunner`**: Executes agent configs without protocol classes (YAML-defined)
- **`VirtualAgentManager`**: Creates and manages virtual agents at runtime

### Observability
- **`TraceService`**: Singleton trace collector with hierarchical runs via `run_context` ContextVar
- **`ExperimentRegistry`**: A/B testing with weighted variant selection

## Code Style

- Python 3.11+
- Protocol-first design (agents are `typing.Protocol` classes)
- Metadata keys: `AGENT_META_KEY`, `TOOL_META_KEY`, `IS_AGENT_INTERFACE`
- Context variables for trace hierarchy (`run_context`)
- Dict dispatch for provider creation in LangChainLLMFactory
- Helper method extraction for low cyclomatic complexity

## Testing

- pytest + pytest-asyncio
- Mock LLM and factory for unit tests (see conftest.py fixtures)
- 23 test modules covering all components
- Target: 90%+ coverage
- Test both sync and async paths, one-shot and react execution

## Boundaries

- Do not modify `_version.py`
- Agent protocols must use `typing.Protocol`
- `@agent` only on Protocol classes
- `@tool` classes need `name` and `description`
- Scanner skips: `pico_ioc`, `pico_agent`, `importlib`, `contextlib`, `pytest`, `_pytest`, `pluggy`
