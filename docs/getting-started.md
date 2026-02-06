# Getting Started

## Installation

Install Pico-Agent using pip:

```bash
pip install pico-agent
```

Install provider extras for your LLM backend:

```bash
pip install pico-agent[openai]      # OpenAI, Azure, DeepSeek, Qwen
pip install pico-agent[anthropic]   # Anthropic Claude
pip install pico-agent[google]      # Google Gemini
pip install pico-agent[all]         # All providers
```

## Basic Setup

### 1. Configure LLM credentials

API keys are configured via `LLMConfig`, provided by a `@factory` in your application:

```python
from pico_ioc import factory, provides
from pico_agent import LLMConfig

@factory
class LLMConfigFactory:
    @provides(LLMConfig)
    def create(self) -> LLMConfig:
        return LLMConfig(
            api_keys={"openai": "sk-..."},
            base_urls={}
        )
```

### 2. Define tools

Tools are `@component` classes decorated with `@tool(name, description)`:

```python
from pico_ioc import component
from pico_agent import tool

@tool(name="search", description="Search the web for information")
@component
class SearchTool:
    def __init__(self, search_service: SearchService):
        self.search_service = search_service

    async def run(self, query: str) -> str:
        return await self.search_service.search(query)
```

### 3. Define agents

Agents are Protocol classes decorated with `@agent(...)`:

```python
from typing import Protocol
from pico_agent import agent, AgentType, AgentCapability

@agent(
    name="research_agent",
    capability=AgentCapability.SMART,
    system_prompt="You are a research assistant. Use the search tool to find information.",
    agent_type=AgentType.REACT,
    tools=["search"],
)
class ResearchAgent(Protocol):
    async def research(self, topic: str) -> str: ...
```

Key `@agent` parameters:

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Unique agent name (required) |
| `capability` | `str` | `AgentCapability` constant — routes to a model |
| `system_prompt` | `str` | System prompt for the LLM |
| `agent_type` | `AgentType` | `ONE_SHOT`, `REACT`, or `WORKFLOW` |
| `tools` | `List[str]` | Tool names to attach |
| `agents` | `List[str]` | Child agent names (multi-agent orchestration) |
| `temperature` | `float` | LLM temperature (default: 0.7) |
| `max_iterations` | `int` | Max ReAct loop iterations (default: 5) |

### 4. Initialize and use

```python
from pico_ioc import init, configuration, DictSource

config = configuration(DictSource({}))
container = init(modules=["myapp"], config=config)

agent = container.get(ResearchAgent)
result = await agent.research("Latest developments in quantum computing")
print(result)
```

## Multi-Agent Orchestration

Agents can delegate to other agents via the `agents` parameter:

```python
@agent(
    name="writer_agent",
    capability=AgentCapability.SMART,
    system_prompt="Write clear, concise content based on research.",
    agent_type=AgentType.ONE_SHOT,
)
class WriterAgent(Protocol):
    async def write(self, topic: str) -> str: ...

@agent(
    name="orchestrator",
    capability=AgentCapability.REASONING,
    system_prompt="Coordinate research and writing tasks.",
    agent_type=AgentType.REACT,
    agents=["research_agent", "writer_agent"],
)
class OrchestratorAgent(Protocol):
    async def process(self, request: str) -> str: ...
```

Child agents are automatically wrapped as tools via `AgentAsTool`, so the orchestrator can invoke them through the LLM's tool-calling mechanism.

## Agent Types

| Type | Description |
|---|---|
| `AgentType.ONE_SHOT` | Single LLM call, no tool loop |
| `AgentType.REACT` | ReAct loop via LangGraph — iterates tools up to `max_iterations` |
| `AgentType.WORKFLOW` | Custom workflow execution |

## Capabilities

Capabilities are mapped to specific models by the `ModelRouter`:

| Capability | Description |
|---|---|
| `AgentCapability.FAST` | Optimized for speed |
| `AgentCapability.SMART` | Balanced performance (default) |
| `AgentCapability.REASONING` | Advanced reasoning |
| `AgentCapability.VISION` | Vision/multimodal support |
| `AgentCapability.CODING` | Code generation |

## Auto-Discovery

Pico-Agent registers itself via the `pico_boot.modules` entry point. When using `pico-boot` or `pico-stack`, the `AgentScanner`, `ToolScanner`, and all infrastructure components are auto-discovered.

## Next Steps

- Read the [Architecture](architecture.md) documentation
- Check the [FAQ](faq.md) for common questions
