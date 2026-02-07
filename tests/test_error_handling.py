from unittest.mock import MagicMock, patch

import pytest

from pico_agent.config import AgentConfig, LLMConfig
from pico_agent.exceptions import AgentConfigurationError, AgentDisabledError, AgentError
from pico_agent.providers import LangChainLLMFactory
from pico_agent.proxy import DynamicAgentProxy
from pico_agent.registry import AgentConfigService, LocalAgentRegistry, ToolRegistry
from pico_agent.router import ModelRouter
from pico_agent.tracing import TraceService


def create_mock_container(with_tracer=False):
    """Create a mock container that behaves like PicoContainer."""
    container = MagicMock()
    tracer = None
    if with_tracer:
        tracer = TraceService()
        container.has.return_value = True
        container.get.return_value = tracer
    else:
        container.has.return_value = False
        container.get.return_value = None
    return container, tracer


class TestProxyErrorHandling:
    """Test that proxy correctly propagates errors without swallowing them."""

    @pytest.fixture
    def setup_proxy(self, mock_llm_factory, mock_central_client, sample_agent_config):
        # Create a fresh local registry for each test
        local_registry = LocalAgentRegistry()
        local_registry.register(
            sample_agent_config.name, type("TestProtocol", (), {"invoke": lambda self, x: x}), sample_agent_config
        )
        config_service = AgentConfigService(mock_central_client, local_registry)
        tool_registry = ToolRegistry()
        model_router = ModelRouter()
        container, tracer = create_mock_container(with_tracer=True)

        proxy = DynamicAgentProxy(
            agent_name="test_agent",
            protocol_cls=type("TestProtocol", (), {"invoke": lambda self, x: x}),
            config_service=config_service,
            tool_registry=tool_registry,
            llm_factory=mock_llm_factory,
            model_router=model_router,
            container=container,
            locator=None,
        )
        return proxy, mock_llm_factory, tracer

    def test_raises_agent_disabled_error(self, setup_proxy):
        proxy, _, _ = setup_proxy
        # Disable the agent at runtime
        proxy.config_service.update_agent_config("test_agent", enabled=False)

        with pytest.raises(AgentDisabledError):
            proxy.invoke("test")

    def test_llm_error_propagates(self, setup_proxy):
        proxy, mock_llm_factory, tracer = setup_proxy
        mock_llm_factory.create.return_value.invoke.side_effect = RuntimeError("LLM crashed")

        with pytest.raises(RuntimeError) as exc_info:
            proxy.invoke("test")

        assert "LLM crashed" in str(exc_info.value)

    def test_error_is_traced(self, setup_proxy):
        proxy, mock_llm_factory, tracer = setup_proxy
        mock_llm_factory.create.return_value.invoke.side_effect = ValueError("Bad input")

        with pytest.raises(ValueError):
            proxy.invoke("test")

        traces = tracer.get_traces()
        assert len(traces) > 0
        assert traces[0]["error"] == "Bad input"

    def test_exception_type_preserved(self, setup_proxy):
        """Ensure using 'raise' instead of 'raise e' preserves exception type."""
        proxy, mock_llm_factory, _ = setup_proxy

        class CustomError(Exception):
            pass

        mock_llm_factory.create.return_value.invoke.side_effect = CustomError("custom")

        with pytest.raises(CustomError):
            proxy.invoke("test")


class TestProviderErrorHandling:
    """Test that providers raise correct error types."""

    def test_missing_api_key_raises_configuration_error(self):
        # Create config without the required key
        config = LLMConfig(api_keys={})
        factory = LangChainLLMFactory(config)

        # Test the require_key function directly by testing create_chat_model
        # with a known provider that will check for API key
        with pytest.raises((AgentConfigurationError, ImportError)) as exc_info:
            # This will either raise AgentConfigurationError (no API key)
            # or ImportError (langchain_openai not installed)
            factory.create("gpt-4", 0.7, 1000)

        # Accept either error - ImportError happens before API key check
        # if langchain_openai isn't installed
        error_str = str(exc_info.value)
        assert "API Key not found" in error_str or "pico-agent[openai]" in error_str

    def test_unknown_provider_raises_value_error(self):
        config = LLMConfig(api_keys={"openai": "key"})
        factory = LangChainLLMFactory(config)

        with pytest.raises(ValueError) as exc_info:
            factory.create_chat_model("unknown_provider", "model", None)

        assert "Unknown LLM Provider" in str(exc_info.value)


class TestChildAgentErrorHandling:
    """Test that child agent resolution logs errors appropriately."""

    def test_disabled_child_agent_is_skipped(self, mock_llm_factory, mock_central_client, sample_agent_config):
        # Setup a parent agent that references a disabled child
        local_registry = LocalAgentRegistry()

        parent_config = AgentConfig(name="parent_agent", system_prompt="Parent", agents=["child_agent"], enabled=True)
        child_config = AgentConfig(
            name="child_agent",
            system_prompt="Child",
            enabled=False,  # Disabled
        )

        class ParentProtocol:
            def invoke(self, x: str) -> str:
                pass

        class ChildProtocol:
            def invoke(self, x: str) -> str:
                pass

        local_registry.register("parent_agent", ParentProtocol, parent_config)
        local_registry.register("child_agent", ChildProtocol, child_config)

        config_service = AgentConfigService(mock_central_client, local_registry)
        tool_registry = ToolRegistry()
        model_router = ModelRouter()
        container, _ = create_mock_container(with_tracer=False)

        # Create a mock locator
        mock_locator = MagicMock()
        mock_child_proxy = MagicMock()
        mock_child_proxy.protocol_cls = ChildProtocol
        mock_locator.get_agent.return_value = mock_child_proxy

        proxy = DynamicAgentProxy(
            agent_name="parent_agent",
            protocol_cls=ParentProtocol,
            config_service=config_service,
            tool_registry=tool_registry,
            llm_factory=mock_llm_factory,
            model_router=model_router,
            container=container,
            locator=mock_locator,
        )

        # Should not raise, child should just be skipped
        tools = proxy._resolve_dependencies(parent_config)

        # Child should not be in tools since it's disabled
        child_names = [getattr(t, "name", None) for t in tools]
        assert "child_agent" not in child_names

    def test_invalid_child_agent_logs_warning(self, mock_llm_factory, mock_central_client, sample_agent_config):
        local_registry = LocalAgentRegistry()

        parent_config = AgentConfig(
            name="parent_agent", system_prompt="Parent", agents=["nonexistent_agent"], enabled=True
        )

        class ParentProtocol:
            def invoke(self, x: str) -> str:
                pass

        local_registry.register("parent_agent", ParentProtocol, parent_config)

        config_service = AgentConfigService(mock_central_client, local_registry)
        tool_registry = ToolRegistry()
        model_router = ModelRouter()
        container, _ = create_mock_container(with_tracer=False)

        # Locator that raises ValueError
        mock_locator = MagicMock()
        mock_locator.get_agent.side_effect = ValueError("Agent not found")

        proxy = DynamicAgentProxy(
            agent_name="parent_agent",
            protocol_cls=ParentProtocol,
            config_service=config_service,
            tool_registry=tool_registry,
            llm_factory=mock_llm_factory,
            model_router=model_router,
            container=container,
            locator=mock_locator,
        )

        # Should not raise, just log warning and continue
        tools = proxy._resolve_dependencies(parent_config)
        assert isinstance(tools, list)


class TestTracingErrorHandling:
    """Test that tracing correctly records errors."""

    def test_trace_service_records_error(self):
        tracer = TraceService()

        run_id = tracer.start_run(name="test_run", run_type="test", inputs={"key": "value"})

        error = ValueError("Something went wrong")
        tracer.end_run(run_id, error=error)

        traces = tracer.get_traces()
        assert len(traces) == 1
        assert traces[0]["error"] == "Something went wrong"
        assert traces[0]["outputs"] is None

    def test_trace_service_records_success(self):
        tracer = TraceService()

        run_id = tracer.start_run(name="success_run", run_type="test", inputs={})

        tracer.end_run(run_id, outputs="Success!")

        traces = tracer.get_traces()
        assert traces[0]["error"] is None
        assert traces[0]["outputs"]["output"] == "Success!"
