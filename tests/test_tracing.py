import pytest
import sys
from unittest.mock import MagicMock
from pico_ioc import init
from pico_agent import agent, AgentCapability, AgentType, TraceService, LLMFactory, LLM
from pico_agent.locator import AgentLocator
from pico_agent.scanner import AgentScanner
from pico_agent.providers import LangChainAdapter
from typing import Protocol
from langchain_core.messages import AIMessage

@agent(
    name="traced_agent", 
    capability=AgentCapability.FAST, 
    agent_type=AgentType.ONE_SHOT,
    system_prompt="You are a tracer"
)
class TracedProto(Protocol):
    def run(self, data: str) -> str: ...

def test_full_observability_flow():
    mock_factory = MagicMock(spec=LLMFactory)
    mock_factory.return_value = mock_factory

    container = init(
        modules=["pico_agent", __name__],
        overrides={LLMFactory: mock_factory}
    )
    
    container.get(AgentScanner).scan_module(sys.modules[__name__])
    tracer = container.get(TraceService)

    mock_chat_model = MagicMock()
    mock_chat_model.invoke.return_value = AIMessage(content="LLM Response")
    mock_chat_model.bind_tools.return_value = mock_chat_model

    adapter_with_tracing = LangChainAdapter(
        chat_model=mock_chat_model, 
        tracer=tracer, 
        model_name="mock-gpt"
    )
    
    mock_factory.create.return_value = adapter_with_tracing

    locator = container.get(AgentLocator)
    agent = locator.get_agent(TracedProto)
    
    result = agent.run("input data")
    
    assert result == "LLM Response"
    
    traces = tracer.get_traces()
    
    assert len(traces) == 2
    
    agent_run = next(t for t in traces if t["run_type"] == "agent")
    llm_run = next(t for t in traces if t["run_type"] == "llm")
    
    assert llm_run["parent_id"] == agent_run["id"]
    
    assert agent_run["inputs"]["data"] == "input data"
    assert llm_run["inputs"]["messages"][1]["content"] == "input data" 
    assert llm_run["outputs"]["output"] == "LLM Response"
