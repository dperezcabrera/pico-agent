import pytest
from unittest.mock import MagicMock
from pico_ioc import init
from pico_agent import VirtualAgentManager, ExperimentRegistry, AgentLocator, LLMFactory, LLM

def test_ab_testing_distribution():
    mock_llm = MagicMock(spec=LLM)
    def side_effect(messages, tools):
        return messages[0]["content"]
    mock_llm.invoke.side_effect = side_effect
    
    mock_factory = MagicMock(spec=LLMFactory)
    mock_factory.create.return_value = mock_llm
    mock_factory.return_value = mock_factory

    container = init(
        modules=["pico_agent"],
        overrides={LLMFactory: mock_factory}
    )

    manager = container.get(VirtualAgentManager)

    manager.create_agent(name="sales_v1", system_prompt="VARIANT_A")
    manager.create_agent(name="sales_v2", system_prompt="VARIANT_B")

    experiments = container.get(ExperimentRegistry)
    experiments.register_experiment(
        public_name="sales_bot",
        variants={
            "sales_v1": 0.5,
            "sales_v2": 0.5
        }
    )

    locator = container.get(AgentLocator)

    counts = {"VARIANT_A": 0, "VARIANT_B": 0}
    
    for _ in range(100):
        agent = locator.get_agent("sales_bot")
        result = agent.run("test")
        counts[result] += 1

    assert counts["VARIANT_A"] > 0
    assert counts["VARIANT_B"] > 0
    assert counts["VARIANT_A"] + counts["VARIANT_B"] == 100

def test_weighted_routing_skew():
    mock_llm = MagicMock(spec=LLM)
    mock_llm.invoke.return_value = "OK"
    
    mock_factory = MagicMock(spec=LLMFactory)
    mock_factory.create.return_value = mock_llm
    mock_factory.return_value = mock_factory

    container = init(
        modules=["pico_agent"],
        overrides={LLMFactory: mock_factory}
    )

    manager = container.get(VirtualAgentManager)
    manager.create_agent(name="canary", system_prompt="NEW")
    manager.create_agent(name="legacy", system_prompt="OLD")

    experiments = container.get(ExperimentRegistry)
    
    experiments.register_experiment(
        public_name="feature_bot",
        variants={"canary": 1.0, "legacy": 0.0}
    )

    locator = container.get(AgentLocator)
    
    for _ in range(10):
        agent = locator.get_agent("feature_bot")
        assert agent.config.name == "canary"
