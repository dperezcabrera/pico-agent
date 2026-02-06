import pytest
from unittest.mock import MagicMock, Mock
from pydantic import BaseModel

from pico_agent.tools import ToolWrapper, AgentAsTool
from pico_agent.config import ToolConfig


class TestToolWrapper:
    @pytest.fixture
    def simple_tool_instance(self):
        class SimpleTool:
            def __call__(self, x: str) -> str:
                return f"processed: {x}"
        return SimpleTool()

    @pytest.fixture
    def run_tool_instance(self):
        class RunTool:
            def run(self, data: str, count: int = 1) -> str:
                return f"ran {count} times with {data}"
        return RunTool()

    @pytest.fixture
    def tool_config(self):
        return ToolConfig(name="test_tool", description="A test tool")

    def test_creates_wrapper_with_call(self, simple_tool_instance, tool_config):
        wrapper = ToolWrapper(simple_tool_instance, tool_config)
        assert wrapper.name == "test_tool"
        assert wrapper.description == "A test tool"

    def test_resolves_call_function(self, simple_tool_instance, tool_config):
        wrapper = ToolWrapper(simple_tool_instance, tool_config)
        assert wrapper.func is not None
        result = wrapper(x="test")
        assert result == "processed: test"

    def test_resolves_run_method(self, run_tool_instance, tool_config):
        wrapper = ToolWrapper(run_tool_instance, tool_config)
        result = wrapper(data="input", count=3)
        assert result == "ran 3 times with input"

    def test_resolves_execute_method(self, tool_config):
        class ExecuteTool:
            def execute(self, value: int) -> int:
                return value * 2

        wrapper = ToolWrapper(ExecuteTool(), tool_config)
        result = wrapper(value=5)
        assert result == 10

    def test_resolves_invoke_method(self, tool_config):
        class InvokeTool:
            def invoke(self, msg: str) -> str:
                return msg.upper()

        wrapper = ToolWrapper(InvokeTool(), tool_config)
        result = wrapper(msg="hello")
        assert result == "HELLO"

    def test_raises_for_missing_method(self, tool_config):
        class NoMethodTool:
            pass

        with pytest.raises(ValueError) as exc_info:
            ToolWrapper(NoMethodTool(), tool_config)
        assert "must implement" in str(exc_info.value)

    def test_creates_args_schema(self, simple_tool_instance, tool_config):
        wrapper = ToolWrapper(simple_tool_instance, tool_config)
        assert wrapper.args_schema is not None
        assert issubclass(wrapper.args_schema, BaseModel)

    def test_args_schema_has_parameters(self, run_tool_instance, tool_config):
        wrapper = ToolWrapper(run_tool_instance, tool_config)
        schema = wrapper.args_schema
        fields = schema.model_fields
        assert "data" in fields
        assert "count" in fields


class TestAgentAsTool:
    @pytest.fixture
    def mock_agent_proxy(self):
        class TestProtocol:
            def invoke(self, message: str) -> str:
                pass

            def process(self, data: str, count: int = 1) -> str:
                pass

        proxy = MagicMock()
        proxy.agent_name = "test_agent"
        proxy.protocol_cls = TestProtocol
        proxy.invoke = MagicMock(return_value="agent result")
        proxy.config_service = None
        return proxy

    def test_creates_agent_as_tool(self, mock_agent_proxy):
        tool = AgentAsTool(mock_agent_proxy)
        assert tool.name == "test_agent"

    def test_uses_provided_description(self, mock_agent_proxy):
        tool = AgentAsTool(mock_agent_proxy, description="Custom description")
        assert tool.description == "Custom description"

    def test_fallback_description(self, mock_agent_proxy):
        tool = AgentAsTool(mock_agent_proxy)
        assert "Agent" in tool.description

    def test_gets_description_from_config_service(self, mock_agent_proxy):
        mock_config_service = MagicMock()
        mock_config = MagicMock()
        mock_config.description = "Agent from config"
        mock_config_service.get_config.return_value = mock_config

        mock_agent_proxy.config_service = mock_config_service

        tool = AgentAsTool(mock_agent_proxy)
        assert tool.description == "Agent from config"

    def test_handles_config_service_error(self, mock_agent_proxy):
        mock_config_service = MagicMock()
        mock_config_service.get_config.side_effect = ValueError("Not found")

        mock_agent_proxy.config_service = mock_config_service

        # Should not raise, should use fallback
        tool = AgentAsTool(mock_agent_proxy)
        assert "Agent" in tool.description

    def test_handles_key_error_from_config(self, mock_agent_proxy):
        mock_config_service = MagicMock()
        mock_config_service.get_config.side_effect = KeyError("missing")

        mock_agent_proxy.config_service = mock_config_service

        tool = AgentAsTool(mock_agent_proxy)
        assert "Agent" in tool.description

    def test_creates_args_schema(self, mock_agent_proxy):
        tool = AgentAsTool(mock_agent_proxy)
        assert tool.args_schema is not None
        assert issubclass(tool.args_schema, BaseModel)

    def test_args_schema_from_protocol(self, mock_agent_proxy):
        tool = AgentAsTool(mock_agent_proxy)
        schema = tool.args_schema
        fields = schema.model_fields
        assert "message" in fields

    def test_call_delegates_to_proxy(self, mock_agent_proxy):
        tool = AgentAsTool(mock_agent_proxy)
        result = tool(message="hello")
        mock_agent_proxy.invoke.assert_called_with(message="hello")

    def test_custom_method_name(self, mock_agent_proxy):
        tool = AgentAsTool(mock_agent_proxy, method_name="process")
        schema = tool.args_schema
        fields = schema.model_fields
        assert "data" in fields
        assert "count" in fields


class TestToolWrapperSchemaGeneration:
    def test_schema_handles_no_params(self):
        class NoParamTool:
            def __call__(self) -> str:
                return "result"

        config = ToolConfig(name="no_param", description="No params")
        wrapper = ToolWrapper(NoParamTool(), config)

        # Should create empty schema
        schema = wrapper.args_schema
        assert len(schema.model_fields) == 0

    def test_schema_handles_optional_params(self):
        class OptionalTool:
            def __call__(self, required: str, optional: int = 10) -> str:
                return f"{required}-{optional}"

        config = ToolConfig(name="optional", description="Optional params")
        wrapper = ToolWrapper(OptionalTool(), config)

        schema = wrapper.args_schema
        fields = schema.model_fields
        assert "required" in fields
        assert "optional" in fields
