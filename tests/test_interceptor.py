import pytest
from unittest.mock import MagicMock, Mock

from pico_agent.interceptor import AgentInterceptor
from pico_agent.proxy import TracedAgentProxy
from pico_agent.decorators import AGENT_META_KEY
from pico_agent.config import AgentConfig


class MockMethodCtx:
    """Mock for pico_ioc MethodCtx."""

    def __init__(self, cls, name, args=(), kwargs=None):
        self.cls = cls
        self.name = name
        self.args = args
        self.kwargs = kwargs or {}


class TestAgentInterceptor:
    @pytest.fixture
    def mock_proxy(self):
        proxy = MagicMock(spec=TracedAgentProxy)
        proxy.execute_agent.return_value = "intercepted result"
        return proxy

    @pytest.fixture
    def interceptor(self, mock_proxy):
        return AgentInterceptor(mock_proxy)

    def test_passes_through_non_agent_classes(self, interceptor):
        class RegularClass:
            pass

        ctx = MockMethodCtx(RegularClass, "some_method", args=("arg1",))
        call_next = MagicMock(return_value="original result")

        result = interceptor.invoke(ctx, call_next)

        assert result == "original result"
        call_next.assert_called_once_with(ctx)

    def test_passes_through_non_invoke_methods(self, interceptor):
        class AgentClass:
            pass

        config = AgentConfig(name="test_agent", system_prompt="Test")
        setattr(AgentClass, AGENT_META_KEY, config)

        ctx = MockMethodCtx(AgentClass, "other_method", args=("arg1",))
        call_next = MagicMock(return_value="other result")

        result = interceptor.invoke(ctx, call_next)

        assert result == "other result"
        call_next.assert_called_once()

    def test_intercepts_invoke_method(self, interceptor, mock_proxy):
        class AgentClass:
            pass

        config = AgentConfig(name="test_agent", system_prompt="Test")
        setattr(AgentClass, AGENT_META_KEY, config)

        ctx = MockMethodCtx(AgentClass, "invoke", args=("user input",))
        call_next = MagicMock()

        result = interceptor.invoke(ctx, call_next)

        assert result == "intercepted result"
        mock_proxy.execute_agent.assert_called_once_with("test_agent", "user input")
        call_next.assert_not_called()

    def test_extracts_input_from_kwargs(self, interceptor, mock_proxy):
        class AgentClass:
            pass

        config = AgentConfig(name="kwarg_agent", system_prompt="Test")
        setattr(AgentClass, AGENT_META_KEY, config)

        ctx = MockMethodCtx(
            AgentClass, "invoke",
            args=(),
            kwargs={"input": "kwarg input"}
        )
        call_next = MagicMock()

        interceptor.invoke(ctx, call_next)

        mock_proxy.execute_agent.assert_called_with("kwarg_agent", "kwarg input")

    def test_extracts_message_from_kwargs(self, interceptor, mock_proxy):
        class AgentClass:
            pass

        config = AgentConfig(name="message_agent", system_prompt="Test")
        setattr(AgentClass, AGENT_META_KEY, config)

        ctx = MockMethodCtx(
            AgentClass, "invoke",
            args=(),
            kwargs={"message": "message input"}
        )
        call_next = MagicMock()

        interceptor.invoke(ctx, call_next)

        mock_proxy.execute_agent.assert_called_with("message_agent", "message input")

    def test_handles_empty_input(self, interceptor, mock_proxy):
        class AgentClass:
            pass

        config = AgentConfig(name="empty_agent", system_prompt="Test")
        setattr(AgentClass, AGENT_META_KEY, config)

        ctx = MockMethodCtx(AgentClass, "invoke", args=(), kwargs={})
        call_next = MagicMock()

        interceptor.invoke(ctx, call_next)

        mock_proxy.execute_agent.assert_called_with("empty_agent", "")

    def test_priority_args_over_kwargs(self, interceptor, mock_proxy):
        """Args should take precedence over kwargs for input."""
        class AgentClass:
            pass

        config = AgentConfig(name="priority_agent", system_prompt="Test")
        setattr(AgentClass, AGENT_META_KEY, config)

        ctx = MockMethodCtx(
            AgentClass, "invoke",
            args=("args_input",),
            kwargs={"input": "kwargs_input"}
        )
        call_next = MagicMock()

        interceptor.invoke(ctx, call_next)

        # Args should be used
        mock_proxy.execute_agent.assert_called_with("priority_agent", "args_input")


class TestAgentInterceptorIntegration:
    def test_interceptor_with_real_proxy(
        self, mock_llm_factory, config_service, tool_registry, model_router
    ):
        proxy = TracedAgentProxy(
            config_service=config_service,
            tool_registry=tool_registry,
            llm_factory=mock_llm_factory,
            model_router=model_router
        )

        interceptor = AgentInterceptor(proxy)

        class MyAgent:
            pass

        config = AgentConfig(name="test_agent", system_prompt="Test")
        setattr(MyAgent, AGENT_META_KEY, config)

        ctx = MockMethodCtx(MyAgent, "invoke", args=("hello",))
        call_next = MagicMock()

        result = interceptor.invoke(ctx, call_next)

        assert result == "mocked response"
