# Frequently Asked Questions

## General

### What is Pico-Agent?

Pico-Agent is a multi-agent orchestration framework that combines pico-ioc's dependency injection with LangChain for LLM integration. It provides declarative agent and tool definitions with automatic discovery, capability-based model routing, and built-in tracing.

### What Python versions are supported?

Pico-Agent requires Python 3.11 or later.

### What LLM providers are supported?

Through LangChain, Pico-Agent supports:
- **OpenAI** -- GPT models (`pico-agent[openai]`)
- **Azure OpenAI** -- Azure-hosted models (`pico-agent[openai]`)
- **Anthropic** -- Claude models (`pico-agent[anthropic]`)
- **Google** -- Gemini models (`pico-agent[google]`)
- **DeepSeek** -- DeepSeek models (`pico-agent[openai]`)
- **Qwen** -- Qwen models (`pico-agent[openai]`)

## Agents

### How do I define an agent?

Use `@agent(...)` on a Protocol class. The `name` parameter is required:

```python
from typing import Protocol
from pico_agent import agent, AgentCapability

@agent(
    name="my_agent",
    capability=AgentCapability.SMART,
    system_prompt="You are a helpful assistant.",
)
class MyAgent(Protocol):
    async def invoke(self, question: str) -> str: ...
```

### What are agent types?

| Type | Behavior |
|---|---|
| `AgentType.ONE_SHOT` | Single LLM call (default) |
| `AgentType.REACT` | ReAct tool loop via LangGraph, iterates up to `max_iterations` |
| `AgentType.WORKFLOW` | Custom workflow execution |

### How do capabilities work?

Capabilities (`AgentCapability.FAST`, `SMART`, `REASONING`, `VISION`, `CODING`) are abstract labels. The `ModelRouter` maps each capability to a concrete model name. This allows changing models globally without modifying agent definitions.

### Can agents call other agents?

Yes, use the `agents` parameter with child agent names:

```python
@agent(
    name="orchestrator",
    capability=AgentCapability.REASONING,
    system_prompt="Coordinate child agents.",
    agent_type=AgentType.REACT,
    agents=["research_agent", "writer_agent"],
)
class OrchestratorAgent(Protocol):
    async def run(self, task: str) -> str: ...
```

Child agents are wrapped as `AgentAsTool` instances so the LLM can invoke them via tool calls.

## Tools

### How do I define a tool?

Use `@tool(name, description)` on a `@component` class:

```python
from pico_ioc import component
from pico_agent import tool

@tool(name="calculator", description="Evaluate math expressions")
@component
class CalculatorTool:
    async def run(self, expression: str) -> str:
        return str(eval(expression))
```

Both `name` and `description` are required. The description is shown to the LLM for tool selection.

### Can tools have dependencies?

Yes, tools are pico-ioc components with full constructor injection:

```python
@tool(name="db_query", description="Query the database")
@component
class DatabaseTool:
    def __init__(self, db_service: DatabaseService):
        self.db = db_service

    async def run(self, query: str) -> str:
        return await self.db.execute(query)
```

### How are tools attached to agents?

By name, via the `tools` parameter in `@agent`:

```python
@agent(name="my_agent", tools=["calculator", "db_query"], ...)
class MyAgent(Protocol):
    ...
```

The `ToolRegistry` resolves tool names to instances at execution time.

## Configuration

### How do I configure API keys?

Pico-Agent registers a default `LLMConfig` singleton automatically. Use `@configure` to populate it with your API keys:

```python
import os
from pico_ioc import component, configure
from pico_agent import LLMConfig

@component
class AppConfig:
    @configure
    def setup_llm(self, config: LLMConfig):
        config.api_keys["openai"] = os.getenv("OPENAI_API_KEY")
        config.api_keys["anthropic"] = os.getenv("ANTHROPIC_API_KEY")
```

**Important:** Do **not** register your own `LLMConfig` with `@factory` + `@provides`.
This would conflict with the singleton already provided by `AgentInfrastructureFactory`.
Always use `@configure` to populate the existing instance.

### Can I use different models for different agents?

Yes, use different `capability` values which the `ModelRouter` maps to different models. You can also use `llm_profile` to select a specific API key/base URL profile.

## Tracing

### How does tracing work?

`TraceService` is a singleton `@component` that captures agent invocations, tool calls, and LLM requests. Tracing is enabled by default (`tracing_enabled=True` in `@agent`). The `DynamicAgentProxy` and `LangChainAdapter` report trace runs automatically.

---

## Troubleshooting

This section documents every error message in pico-agent with its exact text, cause, and fix.

---

### `Agent '<name>' is disabled via configuration.`

**Exception:** `AgentDisabledError`

**Cause:** The agent's `AgentConfig.enabled` field is `False`. This can happen when:

- The agent was explicitly disabled in code (`enabled=False` in `@agent`).
- A `CentralConfigClient` returned a config with `enabled=False`.
- A runtime override set `enabled=False` via `AgentConfigService.update_agent_config()`.

**Fix:** Re-enable the agent:

```python
config_service = container.get(AgentConfigService)
config_service.update_agent_config("my_agent", enabled=True)
```

Or check the central config backend if you use one.

---

### `No configuration found for agent: <name>`

**Exception:** `ValueError`

**Source:** `AgentConfigService.get_config()`

**Cause:** No configuration exists for the agent name in any source (central, local, or runtime).  This usually means:

- The module containing the `@agent`-decorated Protocol was not included in the `init(modules=...)` call.
- The agent name was misspelled.
- The `AgentScanner` did not run (the module was not in the call stack during `@configure`).

**Fix:**

1. Ensure the module is included in the `init()` call:
   ```python
   container = init(modules=["myapp.agents"])
   ```
2. Verify the agent name matches exactly between `@agent(name=...)` and the lookup.
3. Check that the class is decorated with `@agent`, not just defined as a Protocol.

---

### `API Key not found for provider '<provider>' (Profile: '<profile>'). Please configure it via LLMConfig.`

**Exception:** `AgentConfigurationError`

**Source:** `LangChainLLMFactory._require_key()`

**Cause:** The `LLMConfig` singleton does not contain an API key for the detected provider or the specified profile.

**Fix:** Add the key in a `@configure` hook:

```python
@component
class AppConfig:
    @configure
    def setup(self, config: LLMConfig):
        config.api_keys["openai"] = os.getenv("OPENAI_API_KEY")
```

Accepted provider key names: `"openai"`, `"anthropic"`, `"google"`, `"azure"`, `"deepseek"`, `"qwen"`.  If using `llm_profile`, the profile name must also be present in `api_keys`.

---

### `Please install 'pico-agent[openai]' to use this provider.`

**Exception:** `ImportError`

**Source:** `LangChainLLMFactory._create_openai()` (and similar for each provider)

**Cause:** The LangChain package for the target provider is not installed.

**Fix:** Install the correct extra:

```bash
pip install pico-agent[openai]      # OpenAI, Azure, DeepSeek, Qwen
pip install pico-agent[anthropic]   # Claude
pip install pico-agent[google]      # Gemini
pip install pico-agent[all]         # All providers
```

Exact messages per provider:

| Message | Provider |
|---|---|
| `Please install 'pico-agent[openai]' to use this provider.` | OpenAI |
| `Please install 'pico-agent[openai]' to use Azure OpenAI.` | Azure |
| `Please install 'pico-agent[google]' to use Gemini.` | Google |
| `Please install 'pico-agent[anthropic]' to use Claude.` | Anthropic |
| `Please install 'pico-agent[openai]' to use DeepSeek.` | DeepSeek |
| `Please install 'pico-agent[openai]' to use Qwen.` | Qwen |

---

### `Unknown LLM Provider: <provider>`

**Exception:** `ValueError`

**Source:** `LangChainLLMFactory.create_chat_model()`

**Cause:** The provider string (extracted from a `"provider:model"` name or auto-detected) is not one of the supported values.

**Fix:** Use a supported provider name: `openai`, `azure`, `gemini`, `google`, `claude`, `anthropic`, `deepseek`, `qwen`.

---

### `Tool <name> must implement __call__, run, execute, or invoke.`

**Exception:** `ValueError`

**Source:** `ToolWrapper._resolve_function()`

**Cause:** The `@tool`-decorated class does not have any of the expected callable methods.

**Fix:** Add one of `__call__`, `run`, `execute`, or `invoke` to your tool class:

```python
@tool(name="my_tool", description="Does something")
@component
class MyTool:
    def run(self, input: str) -> str:  # <-- add this
        return "result"
```

---

### `Agent <name> has no method <method>`

**Exception:** `AttributeError`

**Source:** `DynamicAgentProxy.__getattr__()`

**Cause:** You called a method on the agent proxy that does not exist on the Protocol class.

**Fix:** Use a method name that is defined on the Protocol:

```python
@agent(name="my_agent", ...)
class MyAgent(Protocol):
    def summarize(self, text: str) -> str: ...  # defined here

agent = locator.get_agent("my_agent")
agent.summarize("Hello")  # correct
# agent.analyze("Hello")  # would raise AttributeError
```

---

### `Virtual Agent '<name>' has no protocol definition.`

**Exception:** `AttributeError`

**Source:** `DynamicAgentProxy.__getattr__()`

**Cause:** The agent proxy has no associated Protocol class.  This happens when a virtual agent (config-only, no Protocol) is accessed via `DynamicAgentProxy` instead of `VirtualAgentRunner`.

**Fix:** Use `VirtualAgentManager` or `AgentLocator.get_agent()` which correctly returns a `VirtualAgentRunner` for virtual agents.

---

### `Cannot call sync run() from inside an async loop. Use await agent.arun() instead.`

**Exception:** `RuntimeError`

**Source:** `VirtualAgentRunner.run_with_args()`

**Cause:** A `WORKFLOW`-type virtual agent's `run()` method was called from inside an already-running asyncio event loop.  Workflow agents use `asyncio.run()` internally, which cannot be nested.

**Fix:** Use the async variant:

```python
# Instead of:
result = agent.run("input")

# Use:
result = await agent.arun("input")
```

---

### `Unknown workflow type: <type>`

**Exception:** `ValueError`

**Source:** `VirtualAgentRunner._arun_workflow()`

**Cause:** The `workflow_config["type"]` value is not recognised.  Currently only `"map_reduce"` is supported.

**Fix:** Set the workflow type to `"map_reduce"`:

```python
config = AgentConfig(
    name="my_workflow",
    agent_type=AgentType.WORKFLOW,
    workflow_config={"type": "map_reduce", "splitter": "...", "reducer": "..."},
)
```

---

### `Map-Reduce requires 'splitter' and 'reducer'`

**Exception:** `ValueError`

**Source:** `VirtualAgentRunner._arun_map_reduce()`

**Cause:** The `workflow_config` dictionary is missing the `"splitter"` or `"reducer"` keys.

**Fix:** Provide both keys:

```python
workflow_config={
    "type": "map_reduce",
    "splitter": "splitter_agent",
    "reducer": "reducer_agent",
    "mapper": "worker_agent",  # or "mappers": {"type_a": "agent_a"}
}
```

---

### `Agent is disabled`

**Exception:** `ValueError`

**Source:** `VirtualAgentRunner.run_structured()`

**Cause:** A structured-output call was made to a disabled virtual agent.

**Fix:** Enable the agent via `AgentConfigService.update_agent_config(name, enabled=True)`.

---

### `Cannot determine module for object <obj>`

**Exception:** `ImportError`

**Source:** `bootstrap._import_module_like()`

**Cause:** An item passed in the `modules` list to `init()` is not a module, string, or object with a `__module__` attribute.

**Fix:** Pass module objects, dotted module name strings, or importable objects:

```python
import myapp
container = init(modules=[myapp])           # module object
container = init(modules=["myapp.agents"])   # string
```

---

### Validation warnings and errors

`AgentValidator.validate()` returns a `ValidationReport` with these possible issues:

| Field | Message | Severity |
|---|---|---|
| `name` | `Agent name cannot be empty` | ERROR |
| `capability` | `Agent capability must be defined` | ERROR |
| `temperature` | `Temperature must be between 0.0 and 2.0` | ERROR |
| `temperature` | `High temperature (>1.0) may cause hallucinations` | WARNING |
| `system_prompt` | `System prompt is empty` | WARNING |

---

### Tools not being called

- Ensure the tool name in `@agent(tools=[...])` matches the `@tool(name=...)` value
- Use `AgentType.REACT` for agents that need to iterate with tools
- Verify the tool description is clear enough for the LLM to select it
- Check that the tool class implements `__call__`, `run`, `execute`, or `invoke`
