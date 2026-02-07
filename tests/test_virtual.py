from unittest.mock import MagicMock

import pytest
from pico_ioc import init

from pico_agent import LLM, AgentCapability, AgentType, LLMFactory, VirtualAgentManager


def test_virtual_agent_lifecycle():
    mock_llm = MagicMock(spec=LLM)
    mock_llm.invoke.return_value = "Virtual Result"

    mock_factory = MagicMock(spec=LLMFactory)
    mock_factory.create.return_value = mock_llm
    mock_factory.return_value = mock_factory

    container = init(modules=["pico_agent"], overrides={LLMFactory: mock_factory})

    manager = container.get(VirtualAgentManager)

    agent_instance = manager.create_agent(
        name="dynamic_bot",
        system_prompt="You are dynamic",
        user_prompt_template="Input: {input}",
        capability=AgentCapability.FAST,
        temperature=0.8,
        agent_type=AgentType.ONE_SHOT,
    )

    response = agent_instance.run("test input")
    assert response == "Virtual Result"

    mock_factory.create.assert_called_with(model_name="gpt-5-mini", temperature=0.8, max_tokens=None, llm_profile=None)

    invoke_args = mock_llm.invoke.call_args[0]
    messages = invoke_args[0]
    assert messages[0]["content"] == "You are dynamic"
    assert messages[1]["content"] == "Input: test input"

    retrieved_agent = manager.get_agent("dynamic_bot")
    assert retrieved_agent is not None
    retrieved_agent.run("another test")

    assert mock_factory.create.call_count == 2
