# 📦 pico-agent

[![PyPI](https://img.shields.io/pypi/v/pico-agent.svg)](https://pypi.org/project/pico-agent/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/dperezcabrera/pico-agent)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![CI (tox matrix)](https://github.com/dperezcabrera/pico-ioc/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/dperezcabrera/pico-agent/branch/main/graph/badge.svg)](https://codecov.io/gh/dperezcabrera/pico-agent)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=dperezcabrera_pico-agent&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=dperezcabrera_pico-agent)
[![Duplicated Lines (%)](https://sonarcloud.io/api/project_badges/measure?project=dperezcabrera_pico-agent&metric=duplicated_lines_density)](https://sonarcloud.io/summary/new_code?id=dperezcabrera_pico-agent)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=dperezcabrera_pico-agent&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=dperezcabrera_pico-agent)


# Pico-Agent

**[Pico-Agent](https://github.com/dperezcabrera/pico-agent)** is a lightweight, protocol-based framework for building modular AI agents using dependency injection.

It eliminates the boilerplate of manually managing LLM chains, prompts, and tools. Instead, you declare **what** you want the agent to do using Python Protocols, and the framework handles **how** to execute it via **[Pico-IoC](https://github.com/dperezcabrera/pico-ioc)**.

> 🤖 **Protocol-First Design**
> 💉 **True Dependency Injection**
> 🧠 **Smart Model Routing**
> 🔄 **Multi-Agent Orchestration**
> 🔌 **LLM Agnostic (LangChain based)**

-----

## 🎯 Why pico-agent?

Most agent frameworks require you to instantiate classes, manually bind tools, and hardcode model names. Pico-Agent moves this complexity into the IoC container, promoting clean architecture and testability.

| Feature | Standard Approach | Pico-Agent |
|----------|-------------------|------------|
| **Definition** | Concrete Classes / Functions | Python Protocols (Interfaces) |
| **Discovery** | Manual Instantiation | Auto-scanning via IoC |
| **Tools** | Manual Binding | DI Injection & Name Reference |
| **Configuration** | Global Env Vars | Injected Configuration Components |
| **Orchestration** | Complex Chains | Agents using Agents as Tools |

-----

## 🧱 Core Features

  - **Declarative Agents:** Use `@agent` on standard `typing.Protocol`.
  - **Capability Routing:** Request `FAST`, `SMART`, `CODING` capabilities instead of specific model names.
  - **Auto-Discovery:** Leverages `pico-ioc` 2.2+ custom scanners to find agents automatically.
  - **Type-Safe:** Full support for Pydantic input/output schemas.
  - **ReAct Loops:** Built-in support for reasoning loops and tool usage.

-----

## 📦 Installation

```bash
pip install pico-agent
```

Install with specific provider support:

```bash
pip install "pico-agent[openai,google]"
```

-----

## 🚀 Quick Example (with pico-stack)

The best way to use `pico-agent` is with `pico-stack`, which handles auto-discovery out of the box.

### 1\. Define the Agent Protocol

The docstring becomes the system prompt. The method signature defines the tool input.

```python
from typing import Protocol
from pico_agent import agent, AgentCapability, AgentType

@agent(
    name="translator",
    capability=AgentCapability.FAST,  # Uses a faster/cheaper model
    system_prompt="You are a professional translator.",
    user_prompt_template="Translate this to Spanish: {text}",
    agent_type=AgentType.ONE_SHOT
)
class TranslatorAgent(Protocol):
    def translate(self, text: str) -> str: ...
```

### 2\. Configure Credentials

Inject the centralized `LLMConfig` and set your keys. This avoids hardcoded environment variable lookups inside the library.

```python
import os
from pico_ioc import component, configure
from pico_agent import LLMConfig

@component
class AppConfig:
    @configure
    def setup_llm(self, config: LLMConfig):
        # Load from Env, Vault, AWS Secrets, etc.
        config.api_keys["openai"] = os.getenv("OPENAI_API_KEY")
        config.api_keys["google"] = os.getenv("GOOGLE_API_KEY")
```

### 3\. Run the Application

```python
import src
from pico_stack import init
from app.agents import TranslatorAgent

def main():
    # Scans modules, finds agents, wires dependencies
    container = init(src)

    # Get your agent instance (auto-generated proxy)
    translator = container.get(TranslatorAgent)

    result = translator.translate(text="Dependency Injection is cool")
    print(result) # "La inyección de dependencias es genial"

if __name__ == "__main__":
    main()
```

-----

## 🛠 Advanced Usage

### ReAct Agents & Tools

Agents can use any dependency in the container as a tool.

```python
# 1. Define a tool (standard component)
from pico_ioc import component

@component
class Calculator:
    def add(self, a: int, b: int) -> int:
        """Adds two numbers."""
        return a + b

# 2. Define the Agent
@agent(
    name="math_expert",
    capability=AgentCapability.REASONING,
    agent_type=AgentType.REACT,  # Enables Reasoning Loop
    tools=["calculator"],        # Reference tool by name (snake_case of class)
    max_iterations=5
)
class MathAgent(Protocol):
    def solve(self, problem: str) -> str: ...
```

### Multi-Agent Orchestration

An agent can use other agents as tools simply by referencing them.

```python
@agent(
    name="orchestrator",
    capability=AgentCapability.SMART,
    system_prompt="Coordinate the sub-agents to solve the task.",
    # Inject other agents as tools available to this LLM
    agents=["translator", "math_expert"] 
)
class Orchestrator(Protocol):
    def handle_request(self, task: str) -> str: ...
```

-----

## ⚙️ Model Routing & Configuration

`pico-agent` uses semantic capabilities to route to specific models. You can configure this globally using a configurer.

```python
from pico_ioc import component, configure
from pico_agent.router import ModelRouter, AgentCapability

@component
class RouterConfig:
    @configure
    def configure_router(self, router: ModelRouter):
        # Update default mappings
        router.update_mapping(AgentCapability.FAST, "openai:gpt-5-mini")
        router.update_mapping(AgentCapability.SMART, "anthropic:claude-4-5-sonnet")
        
        # You can also use "provider:model" syntax
        router.update_mapping(AgentCapability.CODING, "deepseek:deepseek-coder")
```

-----

## 🧪 Testing

Testing is simple because you can mock the underlying `LLMFactory` or the agent protocol itself.

```python
from unittest.mock import MagicMock
from pico_ioc import init
from pico_agent import LLMFactory, LLM

def test_my_agent():
    mock_llm = MagicMock(spec=LLM)
    mock_llm.invoke.return_value = "Mocked Response"
    
    mock_factory = MagicMock(spec=LLMFactory)
    mock_factory.create.return_value = mock_llm
    mock_factory.return_value = mock_factory # For singleton resolution

    container = init(
        modules=["pico_agent", "my_app"],
        overrides={LLMFactory: mock_factory},
        custom_scanners=[AgentScanner()] # If not using pico-stack
    )
    
    agent = container.get(MyAgent)
    assert agent.run("test") == "Mocked Response"
```

-----

## 📝 License

This project is licensed under the MIT License.
