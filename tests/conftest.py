import pytest
from unittest.mock import MagicMock, Mock

from pico_agent.config import AgentConfig, AgentType, AgentCapability, LLMConfig
from pico_agent.interfaces import LLM, LLMFactory, CentralConfigClient
from pico_agent.registry import ToolRegistry, LocalAgentRegistry, AgentConfigService
from pico_agent.router import ModelRouter


@pytest.fixture
def mock_llm():
    """Create a mock LLM that conforms to the LLM protocol."""
    llm = MagicMock(spec=LLM)
    llm.invoke.return_value = "mocked response"
    llm.invoke_structured.return_value = {"result": "structured"}
    llm.invoke_agent_loop.return_value = "agent loop result"
    return llm


@pytest.fixture
def mock_llm_factory(mock_llm):
    """Create a mock LLM factory that returns the mock_llm."""
    factory = MagicMock(spec=LLMFactory)
    factory.create.return_value = mock_llm
    return factory


@pytest.fixture
def mock_central_client():
    """Create a mock CentralConfigClient."""
    client = MagicMock(spec=CentralConfigClient)
    client.get_agent_config.return_value = None
    return client


@pytest.fixture
def sample_agent_config():
    """Create a sample AgentConfig for testing."""
    return AgentConfig(
        name="test_agent",
        system_prompt="You are a helpful test agent.",
        description="A test agent for unit testing",
        capability=AgentCapability.SMART,
        enabled=True,
        agent_type=AgentType.ONE_SHOT,
        tools=["tool1", "tool2"],
        agents=[],
        tags=["test"],
        temperature=0.5,
        max_tokens=1000
    )


@pytest.fixture
def sample_react_config():
    """Create a sample REACT AgentConfig for testing."""
    return AgentConfig(
        name="react_agent",
        system_prompt="You are a ReAct agent.",
        description="A ReAct test agent",
        capability=AgentCapability.REASONING,
        enabled=True,
        agent_type=AgentType.REACT,
        tools=["tool1"],
        agents=["child_agent"],
        tags=["react"],
        temperature=0.3,
        max_iterations=10
    )


@pytest.fixture
def disabled_agent_config():
    """Create a disabled AgentConfig for testing."""
    return AgentConfig(
        name="disabled_agent",
        system_prompt="I am disabled",
        enabled=False
    )


@pytest.fixture
def tool_registry():
    """Create an empty ToolRegistry."""
    return ToolRegistry()


@pytest.fixture
def local_registry():
    """Create an empty LocalAgentRegistry."""
    return LocalAgentRegistry()


@pytest.fixture
def model_router():
    """Create a ModelRouter with default mappings."""
    return ModelRouter()


@pytest.fixture
def config_service(mock_central_client, local_registry, sample_agent_config):
    """Create an AgentConfigService with mocked dependencies."""
    local_registry.register(
        sample_agent_config.name,
        type("TestProtocol", (), {"invoke": lambda self, x: x}),
        sample_agent_config
    )
    return AgentConfigService(mock_central_client, local_registry)


@pytest.fixture
def base_container(mock_llm_factory, mock_central_client):
    """Create a mock container with basic mocked services."""
    container = MagicMock()
    container.has.return_value = False
    container.get.return_value = None
    return container


@pytest.fixture
def llm_config():
    """Create a sample LLMConfig."""
    return LLMConfig(
        api_keys={
            "openai": "test-openai-key",
            "anthropic": "test-anthropic-key",
            "google": "test-google-key"
        },
        base_urls={
            "custom": "https://custom.api.example.com"
        }
    )


class MockModule:
    """A mock module for testing scanner functionality."""
    __name__ = "test_module"


@pytest.fixture
def mock_module():
    """Create a mock module for scanner testing."""
    return MockModule()
