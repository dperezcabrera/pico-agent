from unittest.mock import MagicMock, Mock, patch

import pytest

from pico_agent.config import AgentCapability, AgentConfig, AgentType
from pico_agent.exceptions import AgentDisabledError
from pico_agent.interfaces import LLMFactory
from pico_agent.proxy import DynamicAgentProxy, TracedAgentProxy
from pico_agent.registry import AgentConfigService, LocalAgentRegistry, ToolRegistry
from pico_agent.router import ModelRouter
from pico_agent.tracing import TraceService


def create_mock_container(with_tracer=False):
    """Create a mock container that behaves like PicoContainer."""
    container = MagicMock()
    container.has.return_value = with_tracer
    if with_tracer:
        tracer = TraceService()
        container.get.return_value = tracer
        return container, tracer
    container.get.return_value = None
    return container, None


class TestTracedAgentProxy:
    def test_execute_agent_calls_llm_invoke(
        self, mock_llm_factory, mock_llm, config_service, tool_registry, model_router
    ):
        proxy = TracedAgentProxy(
            config_service=config_service,
            tool_registry=tool_registry,
            llm_factory=mock_llm_factory,
            model_router=model_router,
        )
        result = proxy.execute_agent("test_agent", "Hello")
        mock_llm.invoke.assert_called_once()
        assert result == "mocked response"

    def test_execute_disabled_agent_raises_error(
        self, mock_llm_factory, config_service, tool_registry, model_router, local_registry, disabled_agent_config
    ):
        local_registry.register(disabled_agent_config.name, type("DisabledProtocol", (), {}), disabled_agent_config)
        proxy = TracedAgentProxy(
            config_service=config_service,
            tool_registry=tool_registry,
            llm_factory=mock_llm_factory,
            model_router=model_router,
        )
        with pytest.raises(AgentDisabledError):
            proxy.execute_agent("disabled_agent", "test input")

    def test_execute_react_agent_uses_agent_loop(
        self,
        mock_llm_factory,
        mock_llm,
        tool_registry,
        model_router,
        mock_central_client,
        local_registry,
        sample_react_config,
    ):
        local_registry.register(sample_react_config.name, type("ReactProtocol", (), {}), sample_react_config)
        config_service = AgentConfigService(mock_central_client, local_registry)

        proxy = TracedAgentProxy(
            config_service=config_service,
            tool_registry=tool_registry,
            llm_factory=mock_llm_factory,
            model_router=model_router,
        )
        result = proxy.execute_agent("react_agent", "Process this")
        mock_llm.invoke_agent_loop.assert_called_once()
        assert result == "agent loop result"

    def test_resolves_tools_from_registry(self, mock_llm_factory, config_service, model_router):
        tool_registry = ToolRegistry()
        mock_tool = MagicMock()
        mock_tool.name = "tool1"
        tool_registry.register("tool1", mock_tool)

        proxy = TracedAgentProxy(
            config_service=config_service,
            tool_registry=tool_registry,
            llm_factory=mock_llm_factory,
            model_router=model_router,
        )
        proxy.execute_agent("test_agent", "test")

        call_args = mock_llm_factory.create.return_value.invoke.call_args
        tools_passed = call_args[0][1]
        assert mock_tool in tools_passed


class TestDynamicAgentProxy:
    @pytest.fixture
    def protocol_cls(self):
        class TestProtocol:
            def invoke(self, message: str) -> str:
                pass

            def process(self, data: str, count: int = 1) -> str:
                pass

        return TestProtocol

    @pytest.fixture
    def local_config_service(self, mock_central_client, sample_agent_config):
        """Create a config service with its own local registry to avoid fixture conflicts."""
        registry = LocalAgentRegistry()
        registry.register(
            sample_agent_config.name, type("TestProtocol", (), {"invoke": lambda self, x: x}), sample_agent_config
        )
        return AgentConfigService(mock_central_client, registry)

    @pytest.fixture
    def dynamic_proxy(self, protocol_cls, local_config_service, tool_registry, mock_llm_factory, model_router):
        container, _ = create_mock_container(with_tracer=False)
        return DynamicAgentProxy(
            agent_name="test_agent",
            protocol_cls=protocol_cls,
            config_service=local_config_service,
            tool_registry=tool_registry,
            llm_factory=mock_llm_factory,
            model_router=model_router,
            container=container,
            locator=None,
        )

    def test_getattr_returns_callable(self, dynamic_proxy):
        method = dynamic_proxy.invoke
        assert callable(method)

    def test_getattr_raises_for_private_attrs(self, dynamic_proxy):
        with pytest.raises(AttributeError):
            _ = dynamic_proxy._private

    def test_getattr_raises_for_missing_method(self, dynamic_proxy):
        with pytest.raises(AttributeError):
            _ = dynamic_proxy.nonexistent_method

    def test_method_wrapper_calls_execute(self, dynamic_proxy, mock_llm):
        result = dynamic_proxy.invoke("test message")
        assert result == "mocked response"

    def test_execute_disabled_agent_raises(
        self, protocol_cls, tool_registry, mock_llm_factory, model_router, mock_central_client, disabled_agent_config
    ):
        local_registry = LocalAgentRegistry()
        local_registry.register(disabled_agent_config.name, protocol_cls, disabled_agent_config)
        config_service = AgentConfigService(mock_central_client, local_registry)
        container, _ = create_mock_container(with_tracer=False)

        proxy = DynamicAgentProxy(
            agent_name="disabled_agent",
            protocol_cls=protocol_cls,
            config_service=config_service,
            tool_registry=tool_registry,
            llm_factory=mock_llm_factory,
            model_router=model_router,
            container=container,
            locator=None,
        )
        with pytest.raises(AgentDisabledError):
            proxy.invoke("test")

    def test_extracts_input_context(self, dynamic_proxy):
        import inspect

        from pico_agent.proxy import DynamicAgentProxy

        def sample_method(a: str, b: int = 5):
            pass

        sig = inspect.signature(sample_method)

        context = dynamic_proxy._extract_input_context(sig, ("hello",), {"b": 10})
        assert context["a"] == "hello"
        assert context["b"] == "10"

    def test_builds_messages_with_system_prompt(self, dynamic_proxy):
        from pico_agent.config import AgentConfig
        from pico_agent.messages import build_messages

        config = AgentConfig(
            name="test",
            system_prompt="System prompt here",
        )
        messages = build_messages(config, {"input": "test message"})
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_is_pydantic_model_returns_true_for_pydantic(self, dynamic_proxy):
        from pydantic import BaseModel

        class TestModel(BaseModel):
            value: str

        assert dynamic_proxy._is_pydantic_model(TestModel) is True

    def test_is_pydantic_model_returns_false_for_non_pydantic(self, dynamic_proxy):
        assert dynamic_proxy._is_pydantic_model(str) is False
        assert dynamic_proxy._is_pydantic_model(dict) is False

    def test_proxy_without_protocol_raises_on_getattr(
        self, local_config_service, tool_registry, mock_llm_factory, model_router
    ):
        container, _ = create_mock_container(with_tracer=False)
        proxy = DynamicAgentProxy(
            agent_name="test_agent",
            protocol_cls=None,
            config_service=local_config_service,
            tool_registry=tool_registry,
            llm_factory=mock_llm_factory,
            model_router=model_router,
            container=container,
            locator=None,
        )
        with pytest.raises(AttributeError) as exc_info:
            _ = proxy.invoke
        assert "no protocol definition" in str(exc_info.value)


class TestDynamicAgentProxyWithTracing:
    def test_traces_agent_execution(
        self, mock_central_client, sample_agent_config, tool_registry, mock_llm_factory, model_router
    ):
        local_registry = LocalAgentRegistry()
        local_registry.register(
            sample_agent_config.name, type("TestProtocol", (), {"invoke": lambda self, x: x}), sample_agent_config
        )
        config_service = AgentConfigService(mock_central_client, local_registry)

        tracer = TraceService()
        container = MagicMock()
        container.has.return_value = True
        container.get.return_value = tracer

        class TestProtocol:
            def invoke(self, message: str) -> str:
                pass

        proxy = DynamicAgentProxy(
            agent_name="test_agent",
            protocol_cls=TestProtocol,
            config_service=config_service,
            tool_registry=tool_registry,
            llm_factory=mock_llm_factory,
            model_router=model_router,
            container=container,
            locator=None,
        )

        proxy.invoke("test message")

        traces = tracer.get_traces()
        assert len(traces) > 0
        assert traces[0]["name"] == "test_agent"
        assert traces[0]["run_type"] == "agent"


class TestErrorPropagation:
    def test_exception_preserves_traceback(
        self, mock_central_client, sample_agent_config, tool_registry, mock_llm_factory, model_router
    ):
        """Test that using 'raise' preserves the original traceback."""
        local_registry = LocalAgentRegistry()
        local_registry.register(
            sample_agent_config.name, type("TestProtocol", (), {"invoke": lambda self, x: x}), sample_agent_config
        )
        config_service = AgentConfigService(mock_central_client, local_registry)

        tracer = TraceService()
        container = MagicMock()
        container.has.return_value = True
        container.get.return_value = tracer

        class TestProtocol:
            def invoke(self, message: str) -> str:
                pass

        mock_llm_factory.create.return_value.invoke.side_effect = ValueError("LLM Error")

        proxy = DynamicAgentProxy(
            agent_name="test_agent",
            protocol_cls=TestProtocol,
            config_service=config_service,
            tool_registry=tool_registry,
            llm_factory=mock_llm_factory,
            model_router=model_router,
            container=container,
            locator=None,
        )

        with pytest.raises(ValueError) as exc_info:
            proxy.invoke("test")

        assert "LLM Error" in str(exc_info.value)
        traces = tracer.get_traces()
        assert traces[0]["error"] == "LLM Error"
