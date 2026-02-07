from unittest.mock import MagicMock

import pytest
from pico_ioc import init

from pico_agent import LLM, AgentType, LLMFactory, VirtualAgentManager
from pico_agent.virtual import SplitterOutput, TaskItem


def test_router_map_reduce_with_template_injection():
    mock_llm = MagicMock(spec=LLM)

    def side_effect_structured(messages, tools, schema):
        return SplitterOutput(
            tasks=[
                TaskItem(worker_type="coder", arguments={"filename": "api.py", "specs": "FastAPI endpoint"}),
                TaskItem(worker_type="tester", arguments={"filename": "test_api.py", "target": "api.py"}),
            ]
        )

    def side_effect_invoke(messages, tools):
        user_msg = messages[1]["content"]

        if "Write code" in user_msg:
            return "CODE_GENERATED"
        if "Write tests" in user_msg:
            return "TESTS_GENERATED"
        if "Final Report" in messages[0]["content"]:
            return f"REPORT: {user_msg}"

        return "UNKNOWN"

    mock_llm.invoke_structured.side_effect = side_effect_structured
    mock_llm.invoke.side_effect = side_effect_invoke

    mock_factory = MagicMock(spec=LLMFactory)
    mock_factory.create.return_value = mock_llm
    mock_factory.return_value = mock_factory

    container = init(modules=["pico_agent"], overrides={LLMFactory: mock_factory})

    manager = container.get(VirtualAgentManager)

    manager.create_agent(
        name="python_coder",
        system_prompt="You are a coder.",
        user_prompt_template="Write code for {filename}. Specs: {specs}",
        agent_type=AgentType.ONE_SHOT,
    )

    manager.create_agent(
        name="qa_engineer",
        system_prompt="You are a tester.",
        user_prompt_template="Write tests for {filename} testing {target}",
        agent_type=AgentType.ONE_SHOT,
    )

    manager.create_agent(
        name="tech_lead", system_prompt="Final Report.", user_prompt_template="{input}", agent_type=AgentType.ONE_SHOT
    )

    manager.create_agent(name="pm_splitter", system_prompt="Split tasks.", agent_type=AgentType.ONE_SHOT)

    orchestrator = manager.create_agent(
        name="dev_team",
        agent_type=AgentType.WORKFLOW,
        workflow_config={
            "type": "map_reduce",
            "splitter": "pm_splitter",
            "reducer": "tech_lead",
            "mappers": {"coder": "python_coder", "tester": "qa_engineer"},
        },
    )

    result = orchestrator.run("Build a login feature")

    assert "CODE_GENERATED" in result
    assert "TESTS_GENERATED" in result
    assert result.startswith("REPORT:")
