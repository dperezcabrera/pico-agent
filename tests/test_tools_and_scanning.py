import sys
import pytest
from typing import Protocol
from unittest.mock import MagicMock
from pico_ioc import init
from pico_agent import agent, tool, AgentType
from pico_agent.interfaces import LLMFactory, LLM
from pico_agent.tools import ToolWrapper, AgentAsTool
from pico_agent.locator import AgentLocator
from pico_agent.scanner import AgentScanner, ToolScanner

@tool(name="magic_wand", description="Does magic stuff")
class MagicTool:
    def __call__(self, spell: str) -> str:
        return f"Casting {spell}"

@agent(
    name="wizard",
    tools=["magic_wand"],
    agent_type=AgentType.ONE_SHOT
)
class WizardAgent(Protocol):
    def cast(self, spell: str) -> str: ...

@agent(
    name="apprentice",
    agents=["wizard"]
)
class ApprenticeAgent(Protocol):
    def learn(self, topic: str) -> str: ...

def test_tool_scanning_and_wrapping():
    mock_llm = MagicMock(spec=LLM)
    mock_llm.invoke.return_value = "Result"
    
    mock_factory = MagicMock(spec=LLMFactory)
    mock_factory.create.return_value = mock_llm
    mock_factory.return_value = mock_factory

    container = init(
        modules=["pico_agent", __name__],
        overrides={LLMFactory: mock_factory}
    )

    container.get(AgentScanner).scan_module(sys.modules[__name__])
    container.get(ToolScanner).scan_module(sys.modules[__name__])

    agent_locator = container.get(AgentLocator)
    wizard = agent_locator.get_agent(WizardAgent)
    wizard.cast("fireball")

    tools_passed = mock_llm.invoke.call_args[0][1]
    assert len(tools_passed) == 1
    
    wrapper = tools_passed[0]
    assert isinstance(wrapper, ToolWrapper)
    assert wrapper.name == "magic_wand"
    assert wrapper.description == "Does magic stuff"
    assert wrapper.func("ice") == "Casting ice"
    
    input_schema = wrapper.args_schema
    assert "spell" in input_schema.model_fields

def test_agent_as_tool_resolution():
    mock_llm = MagicMock(spec=LLM)
    mock_llm.invoke.return_value = "Result"
    
    mock_factory = MagicMock(spec=LLMFactory)
    mock_factory.create.return_value = mock_llm
    mock_factory.return_value = mock_factory

    container = init(
        modules=["pico_agent", __name__],
        overrides={LLMFactory: mock_factory}
    )

    container.get(AgentScanner).scan_module(sys.modules[__name__])

    agent_locator = container.get(AgentLocator)
    apprentice = agent_locator.get_agent(ApprenticeAgent)
    apprentice.learn("magic")

    tools_passed = mock_llm.invoke.call_args[0][1]
    assert len(tools_passed) == 1
    
    agent_tool = tools_passed[0]
    assert isinstance(agent_tool, AgentAsTool)
    assert agent_tool.name == "wizard"
