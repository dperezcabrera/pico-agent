import pytest
from typing import Protocol, List, Any, Dict, Optional
from unittest.mock import MagicMock
from pico_ioc import init
from pico_agent import agent, AgentCapability, AgentType
from pico_agent.registry import ToolRegistry
from pico_agent.interfaces import LLMFactory, LLM
from pico_agent.tools import AgentAsTool
from pico_agent.scanner import AgentScanner

@agent(
    name="translator",
    capability=AgentCapability.FAST,
    system_prompt="Translate inputs.",
    user_prompt_template="{text}",
    agent_type=AgentType.ONE_SHOT,
    temperature=1.0
)
class TranslatorAgent(Protocol):
    def translate(self, text: str) -> str: ...

@agent(
    name="researcher",
    capability=AgentCapability.REASONING,
    system_prompt="Research deeply.",
    user_prompt_template="{topic}",
    agent_type=AgentType.REACT,
    max_iterations=3,
    tools=["calculator"]
)
class ResearcherAgent(Protocol):
    def research(self, topic: str) -> str: ...

@agent(
    name="orchestrator",
    capability=AgentCapability.SMART,
    user_prompt_template="Coordinate: {task}",
    agent_type=AgentType.ONE_SHOT,
    agents=["translator", "researcher"]
)
class OrchestratorAgent(Protocol):
    def work(self, task: str) -> str: ...

class CalculatorTool:
    def __repr__(self): return "CalculatorTool"
    def __call__(self, *args, **kwargs): return "42"

def test_full_stack_integration():
    mock_llm = MagicMock(spec=LLM)
    mock_llm.invoke.return_value = "LLM Output"
    mock_llm.invoke_agent_loop.return_value = "Loop Output"

    mock_factory = MagicMock(spec=LLMFactory)
    mock_factory.create.return_value = mock_llm
    
    mock_factory.return_value = mock_factory

    container = init(
        modules=["pico_agent", __name__],
        overrides={
            LLMFactory: mock_factory
        },
        custom_scanners=[AgentScanner()]
    )

    registry = container.get(ToolRegistry)
    registry.register("calculator", CalculatorTool())

    translator = container.get(TranslatorAgent)
    result = translator.translate(text="Hello")

    assert result == "LLM Output"
    
    mock_factory.create.assert_called_with(
        model_name="gpt-5-mini",
        temperature=1.0,
        max_tokens=None,
        llm_profile=None
    )

    researcher = container.get(ResearcherAgent)
    result_loop = researcher.research(topic="Quantum")

    assert result_loop == "Loop Output"

    call_args = mock_factory.create.call_args_list[-1]
    assert call_args.kwargs["model_name"] == "gemini-3-pro"

    loop_call = mock_llm.invoke_agent_loop.call_args
    tools_passed = loop_call[0][1]
    assert len(tools_passed) == 1
    assert isinstance(tools_passed[0], CalculatorTool)

    orchestrator = container.get(OrchestratorAgent)
    orchestrator.work(task="Analyze and Translate")

    invoke_call = mock_llm.invoke.call_args
    tools_passed_orch = invoke_call[0][1]
    
    assert len(tools_passed_orch) == 2
    
    tool_names = [t.name for t in tools_passed_orch if isinstance(t, AgentAsTool)]
    assert "translator" in tool_names
    assert "researcher" in tool_names

    translator_tool = next(t for t in tools_passed_orch if t.name == "translator")
    assert translator_tool.description == "Delegates to agent: translator"
    assert "text" in translator_tool.args_schema.model_fields
