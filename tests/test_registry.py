import pytest
from unittest.mock import MagicMock

from pico_agent.registry import ToolRegistry, LocalAgentRegistry, AgentConfigService
from pico_agent.config import AgentConfig


class TestToolRegistry:
    @pytest.fixture
    def registry(self):
        return ToolRegistry()

    def test_register_and_get_tool(self, registry):
        mock_tool = MagicMock()
        registry.register("my_tool", mock_tool)

        result = registry.get_tool("my_tool")
        assert result is mock_tool

    def test_get_nonexistent_tool_returns_none(self, registry):
        result = registry.get_tool("nonexistent")
        assert result is None

    def test_register_with_tags(self, registry):
        mock_tool = MagicMock()
        registry.register("tagged_tool", mock_tool, tags=["tag1", "tag2"])

        names = registry.get_tool_names_by_tag("tag1")
        assert "tagged_tool" in names

        names = registry.get_tool_names_by_tag("tag2")
        assert "tagged_tool" in names

    def test_get_tool_names_by_nonexistent_tag(self, registry):
        names = registry.get_tool_names_by_tag("nonexistent")
        assert names == []

    def test_get_dynamic_tools_by_agent_tags(self, registry):
        tool1 = MagicMock()
        tool2 = MagicMock()

        registry.register("tool1", tool1, tags=["finance"])
        registry.register("tool2", tool2, tags=["analytics"])

        tools = registry.get_dynamic_tools(["finance"])
        assert tool1 in tools
        assert tool2 not in tools

    def test_get_dynamic_tools_includes_global(self, registry):
        global_tool = MagicMock()
        specific_tool = MagicMock()

        registry.register("global_tool", global_tool, tags=["global"])
        registry.register("specific_tool", specific_tool, tags=["specific"])

        tools = registry.get_dynamic_tools(["other_tag"])
        assert global_tool in tools
        assert specific_tool not in tools

    def test_get_dynamic_tools_no_duplicates(self, registry):
        tool = MagicMock()
        registry.register("multi_tag_tool", tool, tags=["tag1", "tag2", "global"])

        tools = registry.get_dynamic_tools(["tag1", "tag2"])
        assert tools.count(tool) == 1


class TestLocalAgentRegistry:
    @pytest.fixture
    def registry(self):
        return LocalAgentRegistry()

    def test_register_and_get_config(self, registry):
        config = AgentConfig(name="test_agent", system_prompt="Test")

        class TestProtocol:
            pass

        registry.register("test_agent", TestProtocol, config)

        result = registry.get_config("test_agent")
        assert result is config

    def test_register_and_get_protocol(self, registry):
        config = AgentConfig(name="test_agent", system_prompt="Test")

        class TestProtocol:
            pass

        registry.register("test_agent", TestProtocol, config)

        result = registry.get_protocol("test_agent")
        assert result is TestProtocol

    def test_get_nonexistent_config(self, registry):
        result = registry.get_config("nonexistent")
        assert result is None

    def test_get_nonexistent_protocol(self, registry):
        result = registry.get_protocol("nonexistent")
        assert result is None


class TestAgentConfigService:
    @pytest.fixture
    def mock_central_client(self):
        client = MagicMock()
        client.get_agent_config.return_value = None
        return client

    @pytest.fixture
    def local_registry(self):
        return LocalAgentRegistry()

    @pytest.fixture
    def service(self, mock_central_client, local_registry):
        return AgentConfigService(mock_central_client, local_registry)

    def test_get_config_from_local_registry(self, service, local_registry):
        config = AgentConfig(name="local_agent", system_prompt="Local")

        class Protocol:
            pass

        local_registry.register("local_agent", Protocol, config)

        result = service.get_config("local_agent")
        assert result.name == "local_agent"

    def test_get_config_from_central_client(self, service, mock_central_client):
        remote_config = AgentConfig(name="remote_agent", system_prompt="Remote")
        mock_central_client.get_agent_config.return_value = remote_config

        result = service.get_config("remote_agent")
        assert result.name == "remote_agent"
        assert result.system_prompt == "Remote"

    def test_remote_config_takes_precedence(self, service, mock_central_client, local_registry):
        local_config = AgentConfig(name="agent", system_prompt="Local")
        remote_config = AgentConfig(name="agent", system_prompt="Remote")

        class Protocol:
            pass

        local_registry.register("agent", Protocol, local_config)
        mock_central_client.get_agent_config.return_value = remote_config

        result = service.get_config("agent")
        assert result.system_prompt == "Remote"

    def test_raises_for_nonexistent_agent(self, service):
        with pytest.raises(ValueError) as exc_info:
            service.get_config("nonexistent")
        assert "No configuration found" in str(exc_info.value)

    def test_update_agent_config(self, service, local_registry):
        config = AgentConfig(name="agent", system_prompt="Original", temperature=0.5)

        class Protocol:
            pass

        local_registry.register("agent", Protocol, config)

        service.update_agent_config("agent", temperature=0.9)

        result = service.get_config("agent")
        assert result.temperature == 0.9
        assert result.system_prompt == "Original"

    def test_reset_agent_config(self, service, local_registry):
        config = AgentConfig(name="agent", system_prompt="Original", temperature=0.5)

        class Protocol:
            pass

        local_registry.register("agent", Protocol, config)

        service.update_agent_config("agent", temperature=0.9)
        service.reset_agent_config("agent")

        result = service.get_config("agent")
        assert result.temperature == 0.5

    def test_runtime_override_creates_config_if_no_base(self, service):
        service.update_agent_config("new_agent", system_prompt="Runtime created")

        result = service.get_config("new_agent")
        assert result.name == "new_agent"
        assert result.system_prompt == "Runtime created"
