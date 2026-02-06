import pytest
from pico_agent.config import AgentConfig, AgentType, AgentCapability, LLMConfig, ToolConfig


class TestAgentType:
    def test_one_shot_value(self):
        assert AgentType.ONE_SHOT.value == "one_shot"

    def test_react_value(self):
        assert AgentType.REACT.value == "react"

    def test_workflow_value(self):
        assert AgentType.WORKFLOW.value == "workflow"

    def test_is_str_enum(self):
        assert isinstance(AgentType.ONE_SHOT, str)
        assert AgentType.REACT == "react"


class TestAgentCapability:
    def test_fast_capability(self):
        assert AgentCapability.FAST == "fast"

    def test_smart_capability(self):
        assert AgentCapability.SMART == "smart"

    def test_reasoning_capability(self):
        assert AgentCapability.REASONING == "reasoning"

    def test_vision_capability(self):
        assert AgentCapability.VISION == "vision"

    def test_coding_capability(self):
        assert AgentCapability.CODING == "coding"


class TestAgentConfig:
    def test_minimal_config(self):
        config = AgentConfig(name="test")
        assert config.name == "test"
        assert config.system_prompt == ""
        assert config.enabled is True
        assert config.agent_type == AgentType.ONE_SHOT

    def test_full_config(self, sample_agent_config):
        config = sample_agent_config
        assert config.name == "test_agent"
        assert config.system_prompt == "You are a helpful test agent."
        assert config.description == "A test agent for unit testing"
        assert config.capability == AgentCapability.SMART
        assert config.enabled is True
        assert config.tools == ["tool1", "tool2"]
        assert config.temperature == 0.5
        assert config.max_tokens == 1000

    def test_default_values(self):
        config = AgentConfig(name="default_test")
        assert config.user_prompt_template == "{input}"
        assert config.description == ""
        assert config.capability == AgentCapability.SMART
        assert config.enabled is True
        assert config.agent_type == AgentType.ONE_SHOT
        assert config.max_iterations == 5
        assert config.tools == []
        assert config.agents == []
        assert config.tags == []
        assert config.tracing_enabled is True
        assert config.temperature == 0.7
        assert config.max_tokens is None
        assert config.llm_profile is None
        assert config.workflow_config == {}

    def test_react_agent_config(self, sample_react_config):
        config = sample_react_config
        assert config.agent_type == AgentType.REACT
        assert config.max_iterations == 10
        assert config.agents == ["child_agent"]


class TestToolConfig:
    def test_tool_config_creation(self):
        config = ToolConfig(name="my_tool", description="A useful tool")
        assert config.name == "my_tool"
        assert config.description == "A useful tool"


class TestLLMConfig:
    def test_empty_llm_config(self):
        config = LLMConfig()
        assert config.api_keys == {}
        assert config.base_urls == {}

    def test_llm_config_with_keys(self, llm_config):
        assert "openai" in llm_config.api_keys
        assert llm_config.api_keys["openai"] == "test-openai-key"
        assert "custom" in llm_config.base_urls
