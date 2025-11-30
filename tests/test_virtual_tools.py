import pytest
from typing import List, Dict, Any
from unittest.mock import MagicMock
from pico_ioc import init
from pico_agent import VirtualToolManager, DynamicTool, VirtualAgentManager, LLMFactory, LLM

def test_proto_tool_creation_and_use():
    mock_llm = MagicMock(spec=LLM)
    mock_llm.invoke_agent_loop.return_value = "Result from Tool"
    
    mock_factory = MagicMock(spec=LLMFactory)
    mock_factory.create.return_value = mock_llm
    mock_factory.return_value = mock_factory

    container = init(
        modules=["pico_agent"],
        overrides={LLMFactory: mock_factory}
    )

    tool_manager = container.get(VirtualToolManager)
    
    def my_logic(payload: List[Dict[str, Any]]) -> str:
        return ", ".join([str(item.get("val")) for item in payload])

    tool_manager.create_proto_tool(
        name="data_processor",
        description="Processes a list of dictionaries",
        handler=my_logic
    )

    agent_manager = container.get(VirtualAgentManager)
    agent = agent_manager.create_agent(
        name="worker",
        system_prompt="Use the tool",
        tools=["data_processor"],
        agent_type="react"
    )

    agent.run("process this")

    tools_passed = mock_llm.invoke_agent_loop.call_args[0][1]
    assert len(tools_passed) == 1
    
    dynamic_tool = tools_passed[0]
    assert isinstance(dynamic_tool, DynamicTool)
    assert dynamic_tool.name == "data_processor"
    
    result = dynamic_tool(payload=[{"val": "A"}, {"val": "B"}])
    assert result == "A, B"
    
    schema = dynamic_tool.args_schema
    assert "payload" in schema.model_fields
