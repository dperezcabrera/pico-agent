"""Tests for providers.py to improve coverage."""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pico_agent.providers import LangChainAdapter, LangChainLLMFactory
from pico_agent.config import LLMConfig
from pico_agent.exceptions import AgentConfigurationError


class TestLangChainAdapter:
    """Tests for LangChainAdapter."""

    def test_convert_messages_all_roles(self):
        """Converts all message roles correctly."""
        mock_model = MagicMock()
        adapter = LangChainAdapter(mock_model, tracer=None, model_name="test")

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        result = adapter._convert_messages(messages)

        assert len(result) == 3
        assert result[0].__class__.__name__ == "SystemMessage"
        assert result[1].__class__.__name__ == "HumanMessage"
        assert result[2].__class__.__name__ == "AIMessage"

    def test_trace_without_tracer(self):
        """Calls function directly when no tracer."""
        mock_model = MagicMock()
        adapter = LangChainAdapter(mock_model, tracer=None, model_name="test")

        result = adapter._trace("test", [], lambda: "result")

        assert result == "result"

    def test_trace_with_tracer_success(self):
        """Records trace on success."""
        mock_model = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_run.return_value = "run-123"
        adapter = LangChainAdapter(mock_model, tracer=mock_tracer, model_name="test")

        result = adapter._trace("test", [], lambda: "result")

        assert result == "result"
        mock_tracer.start_run.assert_called_once()
        mock_tracer.end_run.assert_called_once_with("run-123", outputs="result")

    def test_trace_with_tracer_error(self):
        """Records trace on error."""
        mock_model = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_run.return_value = "run-123"
        adapter = LangChainAdapter(mock_model, tracer=mock_tracer, model_name="test")

        def raise_error():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            adapter._trace("test", [], raise_error)

        mock_tracer.end_run.assert_called_once()
        call_args = mock_tracer.end_run.call_args
        assert call_args[0][0] == "run-123"
        assert isinstance(call_args[1]["error"], ValueError)

    def test_invoke_with_tools(self):
        """Invoke binds tools when provided."""
        mock_model = MagicMock()
        mock_bound = MagicMock()
        mock_model.bind_tools.return_value = mock_bound
        mock_bound.invoke.return_value = MagicMock(content="response")
        adapter = LangChainAdapter(mock_model, tracer=None, model_name="test")

        result = adapter.invoke([{"role": "user", "content": "hi"}], ["tool1"])

        mock_model.bind_tools.assert_called_once_with(["tool1"])
        assert result == "response"

    def test_invoke_without_tools(self):
        """Invoke works without tools."""
        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content="response")
        adapter = LangChainAdapter(mock_model, tracer=None, model_name="test")

        result = adapter.invoke([{"role": "user", "content": "hi"}], [])

        mock_model.bind_tools.assert_not_called()
        assert result == "response"

    def test_invoke_structured(self):
        """invoke_structured uses with_structured_output."""
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_model.with_structured_output.return_value = mock_structured
        mock_structured.invoke.return_value = {"key": "value"}
        adapter = LangChainAdapter(mock_model, tracer=None, model_name="test")

        class OutputSchema:
            pass

        result = adapter.invoke_structured(
            [{"role": "user", "content": "hi"}], [], OutputSchema
        )

        mock_model.with_structured_output.assert_called_once_with(OutputSchema)
        assert result == {"key": "value"}


class TestLangChainLLMFactory:
    """Tests for LangChainLLMFactory."""

    def test_detect_provider_gemini(self):
        """Detects gemini provider."""
        config = LLMConfig(api_keys={})
        factory = LangChainLLMFactory(config)

        assert factory._detect_provider("gemini-1.5-pro") == "gemini"
        assert factory._detect_provider("GEMINI-FLASH") == "gemini"

    def test_detect_provider_claude(self):
        """Detects claude/anthropic provider."""
        config = LLMConfig(api_keys={})
        factory = LangChainLLMFactory(config)

        assert factory._detect_provider("claude-3-opus") == "claude"
        assert factory._detect_provider("anthropic-model") == "claude"

    def test_detect_provider_deepseek(self):
        """Detects deepseek provider."""
        config = LLMConfig(api_keys={})
        factory = LangChainLLMFactory(config)

        assert factory._detect_provider("deepseek-coder") == "deepseek"

    def test_detect_provider_qwen(self):
        """Detects qwen provider."""
        config = LLMConfig(api_keys={})
        factory = LangChainLLMFactory(config)

        assert factory._detect_provider("qwen-72b") == "qwen"

    def test_detect_provider_azure(self):
        """Detects azure provider."""
        config = LLMConfig(api_keys={})
        factory = LangChainLLMFactory(config)

        assert factory._detect_provider("azure-gpt4") == "azure"

    def test_detect_provider_defaults_to_openai(self):
        """Defaults to openai for unknown models."""
        config = LLMConfig(api_keys={})
        factory = LangChainLLMFactory(config)

        assert factory._detect_provider("gpt-4") == "openai"
        assert factory._detect_provider("unknown-model") == "openai"

    def test_get_api_key_from_profile(self):
        """Gets API key from profile first."""
        config = LLMConfig(api_keys={"myprofile": "profile-key", "openai": "default-key"})
        factory = LangChainLLMFactory(config)

        result = factory._get_api_key("openai", "myprofile")

        assert result == "profile-key"

    def test_get_api_key_from_provider(self):
        """Gets API key from provider when no profile."""
        config = LLMConfig(api_keys={"openai": "provider-key"})
        factory = LangChainLLMFactory(config)

        result = factory._get_api_key("openai", None)

        assert result == "provider-key"

    def test_get_base_url_from_profile(self):
        """Gets base URL from profile first."""
        config = LLMConfig(api_keys={}, base_urls={"myprofile": "http://profile", "openai": "http://default"})
        factory = LangChainLLMFactory(config)

        result = factory._get_base_url("openai", None, "myprofile")

        assert result == "http://profile"

    def test_get_base_url_from_provider(self):
        """Gets base URL from provider when no profile."""
        config = LLMConfig(api_keys={}, base_urls={"openai": "http://provider"})
        factory = LangChainLLMFactory(config)

        result = factory._get_base_url("openai", "http://default", None)

        assert result == "http://provider"

    def test_get_base_url_uses_default(self):
        """Uses default when no base URL configured."""
        config = LLMConfig(api_keys={}, base_urls={})
        factory = LangChainLLMFactory(config)

        result = factory._get_base_url("openai", "http://default", None)

        assert result == "http://default"

    def test_require_key_raises_on_missing(self):
        """Raises AgentConfigurationError when key missing."""
        config = LLMConfig(api_keys={})
        factory = LangChainLLMFactory(config)

        with pytest.raises(AgentConfigurationError, match="API Key not found"):
            factory._require_key("openai", "myprofile")

    def test_create_unknown_provider_raises(self):
        """Raises ValueError for unknown provider."""
        config = LLMConfig(api_keys={})
        factory = LangChainLLMFactory(config)

        with pytest.raises(ValueError, match="Unknown LLM Provider: unknown"):
            factory.create_chat_model("unknown", "model", None)

    def test_create_parses_provider_prefix(self):
        """Parses provider:model format."""
        config = LLMConfig(api_keys={"openai": "test-key"})
        factory = LangChainLLMFactory(config)

        with patch.object(factory, "_create_openai") as mock_create:
            mock_create.return_value = MagicMock()
            factory.create("openai:gpt-4", 0.7, 100)

            mock_create.assert_called_once_with("gpt-4", None, 60)

    def test_create_detects_provider_when_no_prefix(self):
        """Detects provider when no prefix in model name."""
        config = LLMConfig(api_keys={"claude": "test-key"})
        factory = LangChainLLMFactory(config)

        with patch.object(factory, "_detect_provider", return_value="claude"):
            with patch.object(factory, "_create_anthropic") as mock_create:
                mock_create.return_value = MagicMock()
                factory.create("claude-3-opus", 0.7, 100)

                mock_create.assert_called_once()

    def test_create_sets_temperature(self):
        """Sets temperature on model."""
        config = LLMConfig(api_keys={"openai": "test-key"})
        factory = LangChainLLMFactory(config)

        mock_model = MagicMock()
        with patch.object(factory, "_create_openai", return_value=mock_model):
            factory.create("openai:gpt-4", 0.9, None)

            assert mock_model.temperature == 0.9

    def test_create_sets_max_tokens(self):
        """Sets max_tokens on model."""
        config = LLMConfig(api_keys={"openai": "test-key"})
        factory = LangChainLLMFactory(config)

        mock_model = MagicMock()
        with patch.object(factory, "_create_openai", return_value=mock_model):
            factory.create("openai:gpt-4", 0.7, 500)

            assert mock_model.max_tokens == 500

    def test_create_handles_attribute_error_for_temperature(self):
        """Handles AttributeError when model doesn't support temperature."""
        config = LLMConfig(api_keys={"openai": "test-key"})
        factory = LangChainLLMFactory(config)

        mock_model = MagicMock()
        type(mock_model).temperature = PropertyMock(side_effect=AttributeError)
        with patch.object(factory, "_create_openai", return_value=mock_model):
            # Should not raise
            factory.create("openai:gpt-4", 0.9, None)

    def test_create_gets_tracer_from_container(self):
        """Gets TraceService from container if available."""
        config = LLMConfig(api_keys={"openai": "test-key"})
        mock_container = MagicMock()
        mock_container.has.return_value = True
        mock_tracer = MagicMock()
        mock_container.get.return_value = mock_tracer
        factory = LangChainLLMFactory(config, container=mock_container)

        mock_model = MagicMock()
        with patch.object(factory, "_create_openai", return_value=mock_model):
            result = factory.create("openai:gpt-4", 0.7, None)

            assert result.tracer == mock_tracer


class TestProviderCreators:
    """Tests for individual provider creator methods."""

    def test_create_openai_missing_package(self):
        """Raises ImportError when langchain_openai not installed."""
        config = LLMConfig(api_keys={"openai": "test-key"})
        factory = LangChainLLMFactory(config)

        with patch.dict("sys.modules", {"langchain_openai": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                with pytest.raises(ImportError, match="pico-agent\\[openai\\]"):
                    factory._create_openai("gpt-4", None, 60)

    def test_create_azure_missing_package(self):
        """Raises ImportError when azure package not installed."""
        config = LLMConfig(api_keys={"azure": "test-key"})
        factory = LangChainLLMFactory(config)

        with patch.dict("sys.modules", {"langchain_openai": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                with pytest.raises(ImportError, match="Azure OpenAI"):
                    factory._create_azure("gpt-4", None, 60)

    def test_create_gemini_missing_package(self):
        """Raises ImportError when google package not installed."""
        config = LLMConfig(api_keys={"google": "test-key"})
        factory = LangChainLLMFactory(config)

        with patch.dict("sys.modules", {"langchain_google_genai": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                with pytest.raises(ImportError, match="pico-agent\\[google\\]"):
                    factory._create_gemini("gemini-pro", None, 60)

    def test_create_anthropic_missing_package(self):
        """Raises ImportError when anthropic package not installed."""
        config = LLMConfig(api_keys={"anthropic": "test-key"})
        factory = LangChainLLMFactory(config)

        with patch.dict("sys.modules", {"langchain_anthropic": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                with pytest.raises(ImportError, match="pico-agent\\[anthropic\\]"):
                    factory._create_anthropic("claude-3", None, 60)

    def test_create_deepseek_missing_package(self):
        """Raises ImportError when openai package not installed for deepseek."""
        config = LLMConfig(api_keys={"deepseek": "test-key"})
        factory = LangChainLLMFactory(config)

        with patch.dict("sys.modules", {"langchain_openai": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                with pytest.raises(ImportError, match="DeepSeek"):
                    factory._create_deepseek("deepseek-coder", None, 60)

    def test_create_qwen_missing_package(self):
        """Raises ImportError when openai package not installed for qwen."""
        config = LLMConfig(api_keys={"qwen": "test-key"})
        factory = LangChainLLMFactory(config)

        with patch.dict("sys.modules", {"langchain_openai": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                with pytest.raises(ImportError, match="Qwen"):
                    factory._create_qwen("qwen-72b", None, 60)

    def test_create_openai_missing_key(self):
        """Raises AgentConfigurationError when key missing (or ImportError if package missing)."""
        config = LLMConfig(api_keys={})
        factory = LangChainLLMFactory(config)

        # Either raises AgentConfigurationError (key missing) or ImportError (package missing)
        with pytest.raises((AgentConfigurationError, ImportError)):
            factory._create_openai("gpt-4", None, 60)

    def test_create_azure_missing_key(self):
        """Raises AgentConfigurationError when key missing (or ImportError if package missing)."""
        config = LLMConfig(api_keys={})
        factory = LangChainLLMFactory(config)

        with pytest.raises((AgentConfigurationError, ImportError)):
            factory._create_azure("gpt-4", None, 60)

    def test_create_gemini_missing_key(self):
        """Raises AgentConfigurationError when key missing (or ImportError if package missing)."""
        config = LLMConfig(api_keys={})
        factory = LangChainLLMFactory(config)

        with pytest.raises((AgentConfigurationError, ImportError)):
            factory._create_gemini("gemini-pro", None, 60)

    def test_create_anthropic_missing_key(self):
        """Raises AgentConfigurationError when key missing (or ImportError if package missing)."""
        config = LLMConfig(api_keys={})
        factory = LangChainLLMFactory(config)

        with pytest.raises((AgentConfigurationError, ImportError)):
            factory._create_anthropic("claude-3", None, 60)

    def test_create_deepseek_missing_key(self):
        """Raises AgentConfigurationError when key missing (or ImportError if package missing)."""
        config = LLMConfig(api_keys={})
        factory = LangChainLLMFactory(config)

        with pytest.raises((AgentConfigurationError, ImportError)):
            factory._create_deepseek("deepseek-coder", None, 60)

    def test_create_qwen_missing_key(self):
        """Raises AgentConfigurationError when key missing (or ImportError if package missing)."""
        config = LLMConfig(api_keys={})
        factory = LangChainLLMFactory(config)

        with pytest.raises((AgentConfigurationError, ImportError)):
            factory._create_qwen("qwen-72b", None, 60)
