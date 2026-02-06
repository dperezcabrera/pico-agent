# Pico-Agent

Multi-agent orchestration framework built on pico-ioc with LangChain integration.

## Features

- **Agent Definitions**: Define agents as Protocol classes with `@agent` decorator
- **Tool System**: Define tools as `@component` classes with `@tool(name, description)`
- **Multi-Provider LLM**: OpenAI, Anthropic, Google Gemini, Azure, DeepSeek, Qwen via LangChain
- **Agent Types**: `ONE_SHOT`, `REACT` (tool loop via LangGraph), `WORKFLOW`
- **Capabilities**: `FAST`, `SMART`, `REASONING`, `VISION`, `CODING` — auto-routed to models
- **Dependency Injection**: Full constructor-based DI via pico-ioc
- **Tracing**: Built-in `TraceService` for agent/tool/LLM observability
- **Auto-Discovery**: `AgentScanner` and `ToolScanner` via `@configure` hooks

## Quick Start

### Define a tool

```python
from pico_ioc import component
from pico_agent import tool

@tool(name="calculator", description="Perform mathematical calculations")
@component
class CalculatorTool:
    async def run(self, expression: str) -> str:
        return str(eval(expression))
```

### Define an agent

Agents are Protocol classes decorated with `@agent`:

```python
from typing import Protocol
from pico_agent import agent, AgentType, AgentCapability

@agent(
    name="math_agent",
    capability=AgentCapability.SMART,
    system_prompt="You are a helpful math assistant. Use the calculator tool when needed.",
    agent_type=AgentType.REACT,
    tools=["calculator"],
)
class MathAgent(Protocol):
    async def solve(self, problem: str) -> str: ...
```

### Use the agent

```python
from pico_ioc import init

container = init(modules=["myapp"], config=config)
agent = container.get(MathAgent)
result = await agent.solve("What is 15 * 23?")
```

## Installation

```bash
pip install pico-agent
```

Provider extras:

```bash
pip install pico-agent[openai]      # OpenAI / Azure / DeepSeek / Qwen
pip install pico-agent[anthropic]   # Anthropic Claude
pip install pico-agent[google]      # Google Gemini
pip install pico-agent[all]         # All providers
```

## Requirements

- Python 3.11+
- pico-ioc >= 2.2.0
- LangChain >= 0.2.0
- LangGraph

## Documentation

- [Getting Started](getting-started.md) - Installation and basic usage
- [Architecture](architecture.md) - Design and implementation details
- [FAQ](faq.md) - Frequently asked questions

## License

MIT License - see LICENSE file for details.
