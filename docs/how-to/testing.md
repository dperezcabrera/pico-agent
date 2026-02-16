# How to Test Agents

This guide covers strategies for unit testing pico-agent agents, including
mocking LLM responses, creating test fixtures, and testing different agent
types.

## Mocking the LLM

The `LLM` protocol has three methods to mock: `invoke`,
`invoke_structured`, and `invoke_agent_loop`.  Create a simple mock that
returns canned responses:

```python
from unittest.mock import MagicMock
from pico_agent.interfaces import LLM


def create_mock_llm(response: str = "mock response") -> LLM:
    """Create a mock LLM that returns a fixed response."""
    mock = MagicMock(spec=LLM)
    mock.invoke.return_value = response
    mock.invoke_structured.return_value = response
    mock.invoke_agent_loop.return_value = response
    return mock
```

## Mocking the LLMFactory

The `LLMFactory.create()` method returns an `LLM`.  Mock the factory to
return your mock LLM:

```python
from pico_agent.interfaces import LLMFactory


def create_mock_factory(response: str = "mock response") -> LLMFactory:
    """Create a mock LLMFactory that produces mock LLMs."""
    mock_llm = create_mock_llm(response)
    mock_factory = MagicMock(spec=LLMFactory)
    mock_factory.create.return_value = mock_llm
    return mock_factory
```

## pytest Fixtures

Define reusable fixtures in `conftest.py`:

```python
import pytest
from unittest.mock import MagicMock
from pico_agent import (
    AgentConfig,
    AgentCapability,
    AgentType,
    LLMConfig,
    AgentConfigService,
    ToolRegistry,
)
from pico_agent.router import ModelRouter
from pico_agent.proxy import DynamicAgentProxy


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.invoke.return_value = "Test response"
    llm.invoke_structured.return_value = "Structured response"
    llm.invoke_agent_loop.return_value = "React response"
    return llm


@pytest.fixture
def mock_factory(mock_llm):
    factory = MagicMock()
    factory.create.return_value = mock_llm
    return factory


@pytest.fixture
def tool_registry():
    return ToolRegistry()


@pytest.fixture
def model_router():
    return ModelRouter()


@pytest.fixture
def agent_config():
    return AgentConfig(
        name="test_agent",
        capability=AgentCapability.SMART,
        system_prompt="You are a test agent.",
        agent_type=AgentType.ONE_SHOT,
    )
```

## Testing ONE_SHOT Agents

```python
def test_one_shot_agent(mock_llm, mock_factory, agent_config):
    """Test that a ONE_SHOT agent makes a single LLM call."""
    from pico_agent.proxy import DynamicAgentProxy
    from pico_agent.registry import AgentConfigService, LocalAgentRegistry
    from pico_agent.interfaces import CentralConfigClient

    # Setup
    local_registry = LocalAgentRegistry()
    local_registry.register("test_agent", MyAgentProtocol, agent_config)

    central_client = MagicMock(spec=CentralConfigClient)
    central_client.get_agent_config.return_value = None

    config_service = AgentConfigService(central_client, local_registry)
    router = ModelRouter()
    container = MagicMock()
    container.has.return_value = False

    proxy = DynamicAgentProxy(
        agent_name="test_agent",
        protocol_cls=MyAgentProtocol,
        config_service=config_service,
        tool_registry=ToolRegistry(),
        llm_factory=mock_factory,
        model_router=router,
        container=container,
    )

    # Act
    result = proxy.invoke("Hello!")

    # Assert
    assert result == "Test response"
    mock_llm.invoke.assert_called_once()
```

## Testing REACT Agents

```python
def test_react_agent_uses_loop(mock_llm, mock_factory):
    """Test that a REACT agent uses invoke_agent_loop."""
    config = AgentConfig(
        name="react_agent",
        capability=AgentCapability.SMART,
        system_prompt="Use tools to answer.",
        agent_type=AgentType.REACT,
        max_iterations=3,
        tools=["calculator"],
    )

    # ... setup similar to above ...

    result = proxy.invoke("What is 2 + 2?")

    mock_llm.invoke_agent_loop.assert_called_once()
    # Verify max_iterations was passed
    call_args = mock_llm.invoke_agent_loop.call_args
    assert call_args[0][2] == 3  # max_iterations
```

## Testing Structured Output

```python
from pydantic import BaseModel


class AnalysisResult(BaseModel):
    summary: str
    confidence: float


def test_structured_output(mock_llm, mock_factory):
    """Test that Pydantic return types trigger structured output."""
    expected = AnalysisResult(summary="Test", confidence=0.95)
    mock_llm.invoke_structured.return_value = expected

    # Define a protocol with Pydantic return type
    @agent(name="analyzer", system_prompt="Analyze text.")
    class Analyzer(Protocol):
        def analyze(self, text: str) -> AnalysisResult: ...

    # ... setup proxy ...

    result = proxy.analyze("Some text")
    assert isinstance(result, AnalysisResult)
    mock_llm.invoke_structured.assert_called_once()
```

## Testing Tools

```python
def test_tool_wrapper():
    """Test that ToolWrapper correctly wraps a tool instance."""
    from pico_agent.tools import ToolWrapper
    from pico_agent.config import ToolConfig

    @tool(name="echo", description="Echoes input")
    class EchoTool:
        def run(self, text: str) -> str:
            return text

    instance = EchoTool()
    config = ToolConfig(name="echo", description="Echoes input")
    wrapper = ToolWrapper(instance, config)

    assert wrapper.name == "echo"
    assert wrapper("text") == "text"  # Note: uses __call__
```

## Testing Virtual Agents

```python
def test_virtual_agent(mock_factory):
    """Test a virtual agent created at runtime."""
    from pico_agent.virtual import VirtualAgentRunner

    config = AgentConfig(
        name="virtual_test",
        system_prompt="You are helpful.",
        capability=AgentCapability.FAST,
    )

    runner = VirtualAgentRunner(
        config=config,
        tool_registry=ToolRegistry(),
        llm_factory=mock_factory,
        model_router=ModelRouter(),
        container=MagicMock(),
        locator=MagicMock(),
        scheduler=MagicMock(),
    )

    result = runner.run("Hello!")
    assert result is not None
```

## Testing Async Agents

Use `pytest-asyncio` for async tests:

```python
import pytest


@pytest.mark.asyncio
async def test_async_agent(mock_factory):
    """Test async agent execution."""
    # ... setup ...

    result = await proxy.arun("Hello!")
    assert result is not None
```

## Testing with the Full Container

For integration tests, use `pico_agent.init()` with mock modules:

```python
def test_full_container():
    """Integration test with a real container."""
    import myapp
    from pico_agent import init

    container = init(modules=[myapp])

    # Override the LLM factory with a mock
    mock_factory = create_mock_factory("Integration test response")
    # Use container overrides or test-specific config
```

## Testing Configuration Merging

```python
def test_config_priority():
    """Test that central config overrides local config."""
    local_config = AgentConfig(name="agent", temperature=0.7)
    central_config = AgentConfig(name="agent", temperature=0.3)

    local_registry = LocalAgentRegistry()
    local_registry.register("agent", MagicMock(), local_config)

    central_client = MagicMock()
    central_client.get_agent_config.return_value = central_config

    service = AgentConfigService(central_client, local_registry)
    result = service.get_config("agent")

    assert result.temperature == 0.3  # Central wins
```

## Testing Validation

```python
def test_validator_catches_errors():
    """Test that AgentValidator catches invalid configs."""
    from pico_agent import AgentValidator

    validator = AgentValidator()

    config = AgentConfig(name="", capability="smart")
    report = validator.validate(config)

    assert not report.valid
    assert report.has_errors
    assert any(i.field == "name" for i in report.issues)
```
