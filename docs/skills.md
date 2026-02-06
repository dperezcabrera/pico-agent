# Claude Code Skills

Pico-Agent includes pre-designed skills for [Claude Code](https://claude.ai/claude-code) that enable AI-assisted development following pico-framework patterns and best practices.

## Available Skills

| Skill | Command | Description |
|-------|---------|-------------|
| **Pico Agent Creator** | `/pico-agent` | Creates LLM agents with tools, prompts and configuration |
| **Pico Test Generator** | `/pico-tests` | Generates tests for pico-framework components |

---

## Pico Agent Creator

Creates agents with protocol-based design, capability routing, and tool integration.

### Agent with Decorator

```python
from pico_agent import agent, AgentType, AgentCapability

@agent(
    name="translator",
    capability=AgentCapability.SMART,  # FAST, SMART, REASONING
    system_prompt="You are a professional translator.",
    agent_type=AgentType.ONE_SHOT,  # ONE_SHOT, REACT
    tools=["dictionary"],
    temperature=0.7,
)
class TranslatorAgent(Protocol):
    def translate(self, text: str) -> str: ...
```

### Virtual Agent (YAML)

```yaml
# agents/translator.yaml
name: translator
capability: smart
agent_type: one_shot
system_prompt: |
  You are a professional translator.
tools:
  - dictionary
temperature: 0.7
```

### Custom Tool

```python
from pico_agent import tool

@tool(name="dictionary", description="Looks up word definitions")
class DictionaryTool:
    def __init__(self, service: DictService):
        self.service = service

    def __call__(self, word: str) -> str:
        return self.service.lookup(word)
```

### Multi-Agent Orchestration

```python
@agent(
    name="orchestrator",
    capability=AgentCapability.SMART,
    system_prompt="Coordinate sub-agents to solve the task.",
    agents=["translator", "math_expert"]
)
class Orchestrator(Protocol):
    def handle_request(self, task: str) -> str: ...
```

---

## Pico Test Generator

Generates tests for any pico-framework component.

### Testing Agents

```python
import pytest
from unittest.mock import MagicMock
from pico_ioc import init
from pico_agent import LLMFactory, LLM

def test_my_agent():
    mock_llm = MagicMock(spec=LLM)
    mock_llm.invoke.return_value = "Mocked Response"

    mock_factory = MagicMock(spec=LLMFactory)
    mock_factory.create.return_value = mock_llm

    container = init(
        modules=["pico_agent", "my_app"],
        overrides={LLMFactory: mock_factory},
    )
    agent = container.get(MyAgent)
    assert agent.run("test") == "Mocked Response"
```

---

## Installation

```bash
# Project-level (recommended)
mkdir -p .claude/skills/pico-agent
# Copy the skill YAML+Markdown to .claude/skills/pico-agent/SKILL.md

mkdir -p .claude/skills/pico-tests
# Copy the skill YAML+Markdown to .claude/skills/pico-tests/SKILL.md

# Or user-level (available in all projects)
mkdir -p ~/.claude/skills/pico-agent
mkdir -p ~/.claude/skills/pico-tests
```

## Usage

```bash
# Invoke directly in Claude Code
/pico-agent translator
/pico-tests TranslatorAgent
```

See the full skill templates in the [pico-framework skill catalog](https://github.com/dperezcabrera/pico-agent).
