"""Tests for locator.py edge cases and coverage."""
import pytest
from unittest.mock import MagicMock, patch
from pico_agent.locator import AgentLocator, NoOpCentralClient, AgentInfrastructureFactory
from pico_agent.config import AgentConfig


class TestNoOpCentralClient:
    """Tests for NoOpCentralClient."""

    def test_get_agent_config_returns_none(self):
        """Always returns None."""
        client = NoOpCentralClient()
        assert client.get_agent_config("any") is None

    def test_upsert_agent_config_does_nothing(self):
        """Does nothing, no error."""
        client = NoOpCentralClient()
        config = MagicMock()
        # Should not raise
        client.upsert_agent_config(config)


class TestAgentInfrastructureFactory:
    """Tests for AgentInfrastructureFactory."""

    def test_provide_llm_factory_creates_factory(self):
        """Creates LangChainLLMFactory with container."""
        from pico_agent.config import LLMConfig
        from pico_agent.providers import LangChainLLMFactory

        mock_container = MagicMock()
        factory = AgentInfrastructureFactory(mock_container)
        config = LLMConfig()

        result = factory.provide_llm_factory(config)

        assert isinstance(result, LangChainLLMFactory)
        assert result.container == mock_container


class TestAgentLocator:
    """Tests for AgentLocator."""

    def _create_locator(self):
        """Create locator with mocked dependencies."""
        return AgentLocator(
            container=MagicMock(),
            config_service=MagicMock(),
            tool_registry=MagicMock(),
            llm_factory=MagicMock(),
            local_registry=MagicMock(),
            model_router=MagicMock(),
            experiment_registry=MagicMock(),
            scheduler=MagicMock(),
        )

    def test_get_agent_by_name_string(self):
        """Gets agent by string name."""
        locator = self._create_locator()
        locator.experiment_registry.resolve_variant.return_value = "my_agent"
        locator.local_registry.get_protocol.return_value = MagicMock()

        result = locator.get_agent("my_agent")

        assert result is not None
        locator.experiment_registry.resolve_variant.assert_called_once_with("my_agent")

    def test_get_agent_by_protocol_with_meta(self):
        """Gets agent by protocol type with AGENT_META_KEY."""
        from pico_agent.decorators import AGENT_META_KEY

        class MockProtocol:
            pass

        mock_meta = MagicMock()
        mock_meta.name = "test_agent"
        setattr(MockProtocol, AGENT_META_KEY, mock_meta)

        locator = self._create_locator()
        result = locator.get_agent(MockProtocol)

        assert result is not None

    def test_get_agent_by_protocol_without_meta_searches_registry(self):
        """Searches local registry when protocol has no meta."""

        class MockProtocol:
            pass

        locator = self._create_locator()
        locator.local_registry._protocols = {"found_agent": MockProtocol}

        result = locator.get_agent(MockProtocol)

        assert result is not None

    def test_get_agent_by_protocol_not_found_returns_none(self):
        """Returns None when protocol not found in registry."""

        class MockProtocol:
            pass

        locator = self._create_locator()
        locator.local_registry._protocols = {}

        result = locator.get_agent(MockProtocol)

        assert result is None

    def test_get_agent_returns_none_when_no_agent_name(self):
        """Returns None when agent name cannot be determined."""
        locator = self._create_locator()
        locator.experiment_registry.resolve_variant.return_value = ""
        locator.local_registry.get_protocol.return_value = None

        result = locator.get_agent("unknown")

        assert result is None

    def test_get_agent_creates_virtual_runner_when_no_protocol(self):
        """Creates VirtualAgentRunner when config exists but no protocol."""
        locator = self._create_locator()
        locator.experiment_registry.resolve_variant.return_value = "virtual_agent"
        locator.local_registry.get_protocol.return_value = None
        locator.config_service.get_config.return_value = MagicMock()

        result = locator.get_agent("virtual_agent")

        assert result is not None
        assert result.__class__.__name__ == "VirtualAgentRunner"

    def test_get_agent_handles_value_error(self):
        """Returns None when config_service raises ValueError."""
        locator = self._create_locator()
        locator.experiment_registry.resolve_variant.return_value = "missing_agent"
        locator.local_registry.get_protocol.return_value = None
        locator.config_service.get_config.side_effect = ValueError("Not found")

        result = locator.get_agent("missing_agent")

        assert result is None

    def test_create_proxy_uses_protocol_meta(self):
        """create_proxy extracts name from protocol meta."""
        from pico_agent.decorators import AGENT_META_KEY

        class TestProtocol:
            pass

        mock_meta = MagicMock()
        mock_meta.name = "protocol_agent"
        setattr(TestProtocol, AGENT_META_KEY, mock_meta)

        locator = self._create_locator()
        result = locator.create_proxy(TestProtocol)

        assert result is not None
        assert result.agent_name == "protocol_agent"
