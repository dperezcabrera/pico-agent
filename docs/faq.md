# Frequently Asked Questions

## General

### What is Pico-Agent?

Pico-Agent is a multi-agent orchestration framework that combines pico-ioc's dependency injection with LangChain for LLM integration. It provides declarative agent and tool definitions with automatic discovery, capability-based model routing, and built-in tracing.

### What Python versions are supported?

Pico-Agent requires Python 3.11 or later.

### What LLM providers are supported?

Through LangChain, Pico-Agent supports:
- **OpenAI** — GPT models (`pico-agent[openai]`)
- **Azure OpenAI** — Azure-hosted models (`pico-agent[openai]`)
- **Anthropic** — Claude models (`pico-agent[anthropic]`)
- **Google** — Gemini models (`pico-agent[google]`)
- **DeepSeek** — DeepSeek models (`pico-agent[openai]`)
- **Qwen** — Qwen models (`pico-agent[openai]`)

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

### Can I use different models for different agents?

Yes, use different `capability` values which the `ModelRouter` maps to different models. You can also use `llm_profile` to select a specific API key/base URL profile.

## Tracing

### How does tracing work?

`TraceService` is a singleton `@component` that captures agent invocations, tool calls, and LLM requests. Tracing is enabled by default (`tracing_enabled=True` in `@agent`). The `DynamicAgentProxy` and `LangChainAdapter` report trace runs automatically.

## Troubleshooting

### Agent not found

Ensure the module containing your agent is scanned by pico-ioc:

```python
container = init(modules=["myapp.agents"], config=config)
```

### Missing provider dependency

Install the correct extra for your LLM provider:

```bash
pip install pico-agent[openai]      # OpenAI, Azure, DeepSeek, Qwen
pip install pico-agent[anthropic]   # Claude
pip install pico-agent[google]      # Gemini
```

### API Key not found

Ensure your `LLMConfig` includes the key for the provider. The key name must match the provider: `"openai"`, `"anthropic"`, `"google"`, `"azure"`, `"deepseek"`, `"qwen"`.

### Tools not being called

- Ensure the tool name in `@agent(tools=[...])` matches the `@tool(name=...)` value
- Use `AgentType.REACT` for agents that need to iterate with tools
- Verify the tool description is clear enough for the LLM to select it
