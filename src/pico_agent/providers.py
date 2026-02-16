"""LangChain-based LLM adapter and multi-provider factory.

``LangChainAdapter`` implements the ``LLM`` protocol by delegating to a
LangChain ``BaseChatModel``.  ``LangChainLLMFactory`` implements the
``LLMFactory`` protocol and creates adapters for OpenAI, Azure, Anthropic,
Google/Gemini, DeepSeek, and Qwen providers.
"""

from typing import Any, Dict, List, Optional, Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .config import LLMConfig
from .exceptions import AgentConfigurationError
from .interfaces import LLM, LLMFactory
from .logging import get_logger

logger = get_logger(__name__)


class LangChainAdapter(LLM):
    """Adapts a LangChain ``BaseChatModel`` to the pico-agent ``LLM`` protocol.

    Handles message conversion, tool binding, structured output, and the
    ReAct agent loop via LangGraph.  Optionally records traces through
    ``TraceService``.

    Args:
        chat_model: A LangChain chat model instance.
        tracer: Optional ``TraceService`` for recording LLM invocations.
        model_name: Human-readable model identifier used in trace names.
    """

    def __init__(self, chat_model: BaseChatModel, tracer: Any = None, model_name: str = ""):
        self.model = chat_model
        self.tracer = tracer
        self.model_name = model_name

    def _convert_messages(self, messages: List[Dict[str, str]]) -> List[BaseMessage]:
        """Convert pico-agent message dicts to LangChain ``BaseMessage`` objects.

        Args:
            messages: List of dicts with ``"role"`` and ``"content"`` keys.

        Returns:
            List of ``SystemMessage``, ``HumanMessage``, or ``AIMessage``.
        """
        lc_messages = []
        for msg in messages:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))
        return lc_messages

    def _trace(self, method_name, inputs, func):
        run_id = None
        if self.tracer:
            run_id = self.tracer.start_run(
                name=f"LLM: {self.model_name}",
                run_type="llm",
                inputs={"messages": inputs},
                extra={"method": method_name},
            )
        try:
            result = func()
            if self.tracer and run_id:
                self.tracer.end_run(run_id, outputs=str(result))
            return result
        except Exception as e:
            if self.tracer and run_id:
                self.tracer.end_run(run_id, error=e)
            raise

    def invoke(self, messages: List[Dict[str, str]], tools: List[Any]) -> str:
        """Send messages to the LLM and return the text response.

        Args:
            messages: List of message dicts with ``"role"`` and ``"content"``.
            tools: LangChain-compatible tool instances to bind.

        Returns:
            The model's text response.
        """

        def _exec():
            lc_messages = self._convert_messages(messages)
            model_with_tools = self.model
            if tools:
                model_with_tools = self.model.bind_tools(tools)
            response = model_with_tools.invoke(lc_messages)
            return str(response.content)

        return self._trace("invoke", messages, _exec)

    def invoke_structured(self, messages: List[Dict[str, str]], tools: List[Any], output_schema: Type[Any]) -> Any:
        """Send messages and parse the response into a Pydantic model.

        Args:
            messages: List of message dicts.
            tools: LangChain-compatible tool instances.
            output_schema: A ``pydantic.BaseModel`` subclass.

        Returns:
            An instance of *output_schema* populated from the LLM response.
        """

        def _exec():
            lc_messages = self._convert_messages(messages)
            structured_model = self.model.with_structured_output(output_schema)
            return structured_model.invoke(lc_messages)

        return self._trace("invoke_structured", messages, _exec)

    def invoke_agent_loop(
        self,
        messages: List[Dict[str, str]],
        tools: List[Any],
        max_iterations: int,
        output_schema: Optional[Type[Any]] = None,
    ) -> Any:
        """Run a ReAct-style tool loop via LangGraph.

        Args:
            messages: List of message dicts.
            tools: LangChain-compatible tool instances.
            max_iterations: Maximum reasoning iterations (``recursion_limit``).
            output_schema: Optional Pydantic model for structured final output.

        Returns:
            The final text response, or an *output_schema* instance if provided.
        """

        def _exec():
            from langgraph.prebuilt import create_react_agent

            lc_messages = self._convert_messages(messages)
            agent_executor = create_react_agent(self.model, tools=tools)
            inputs = {"messages": lc_messages}
            result = agent_executor.invoke(inputs, config={"recursion_limit": max_iterations})
            final_message = result["messages"][-1]

            if output_schema:
                structured_model = self.model.with_structured_output(output_schema)
                return structured_model.invoke([HumanMessage(content=str(final_message.content))])
            return str(final_message.content)

        return self._trace("invoke_agent_loop", messages, _exec)


class LangChainLLMFactory(LLMFactory):
    """Multi-provider LLM factory backed by LangChain chat model classes.

    Supports OpenAI, Azure, Anthropic (Claude), Google (Gemini), DeepSeek,
    and Qwen.  Provider detection is automatic based on the model name, or
    explicit via the ``"provider:model"`` syntax (e.g., ``"openai:gpt-5-mini"``).

    Args:
        config: ``LLMConfig`` containing API keys and base URLs.
        container: Optional ``PicoContainer`` used to resolve ``TraceService``.

    Raises:
        AgentConfigurationError: If the required API key is missing.
        ImportError: If the required LangChain provider package is not
            installed.
        ValueError: If the provider name is unknown.
    """

    def __init__(self, config: LLMConfig, container: Any = None):
        self.config = config
        self.container = container

    def create(
        self, model_name: str, temperature: float, max_tokens: Optional[int], llm_profile: Optional[str] = None
    ) -> LLM:
        """Create a ``LangChainAdapter`` for the given model.

        The provider is detected from the model name or specified explicitly
        using the ``"provider:model"`` syntax.

        Args:
            model_name: Model identifier, optionally prefixed with provider
                (e.g., ``"openai:gpt-5-mini"``).
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens, or ``None``.
            llm_profile: Named profile for API-key / base-URL selection.

        Returns:
            A configured ``LangChainAdapter`` instance.

        Raises:
            AgentConfigurationError: If the API key for the detected provider
                is missing.  Message pattern:
                ``"API Key not found for provider '<provider>' (Profile: '<profile>'). Please configure it via LLMConfig."``
            ImportError: If the LangChain package for the provider is not
                installed.  Message pattern:
                ``"Please install 'pico-agent[<extra>]' to use <provider>."``
            ValueError: If the provider name is not recognised.  Message:
                ``"Unknown LLM Provider: <provider>"``
        """
        final_provider = None
        real_model_name = model_name

        if ":" in model_name:
            parts = model_name.split(":", 1)
            final_provider = parts[0]
            real_model_name = parts[1]

        if not final_provider:
            final_provider = self._detect_provider(real_model_name)

        chat_model = self.create_chat_model(final_provider, real_model_name, llm_profile)

        if temperature is not None:
            try:
                chat_model.temperature = temperature
            except AttributeError:
                pass

        if max_tokens is not None:
            try:
                chat_model.max_tokens = max_tokens
            except AttributeError:
                pass

        tracer = None
        if self.container:
            try:
                from .tracing import TraceService

                if self.container.has(TraceService):
                    tracer = self.container.get(TraceService)
            except ImportError:
                pass

        return LangChainAdapter(chat_model, tracer, real_model_name)

    def _get_api_key(self, provider: str, profile: Optional[str]) -> Optional[str]:
        if profile and profile in self.config.api_keys:
            return self.config.api_keys[profile]
        return self.config.api_keys.get(provider)

    def _get_base_url(self, provider: str, default: Optional[str], profile: Optional[str]) -> Optional[str]:
        if profile and profile in self.config.base_urls:
            return self.config.base_urls[profile]
        return self.config.base_urls.get(provider, default)

    def _detect_provider(self, model_name: str) -> str:
        """Auto-detect the LLM provider from a model name.

        Detection rules (first match wins):

        - Contains ``"gemini"`` -> ``"gemini"``
        - Contains ``"claude"`` or ``"anthropic"`` -> ``"claude"``
        - Contains ``"deepseek"`` -> ``"deepseek"``
        - Contains ``"qwen"`` -> ``"qwen"``
        - Contains ``"azure"`` -> ``"azure"``
        - Otherwise -> ``"openai"``

        Args:
            model_name: The model identifier to inspect.

        Returns:
            A provider key string.
        """
        name_lower = model_name.lower()
        if "gemini" in name_lower:
            return "gemini"
        elif "claude" in name_lower or "anthropic" in name_lower:
            return "claude"
        elif "deepseek" in name_lower:
            return "deepseek"
        elif "qwen" in name_lower:
            return "qwen"
        elif "azure" in name_lower:
            return "azure"
        return "openai"

    def _require_key(self, provider_name: str, profile: Optional[str]) -> str:
        """Get API key or raise configuration error."""
        key = self._get_api_key(provider_name, profile)
        if not key:
            raise AgentConfigurationError(
                f"API Key not found for provider '{provider_name}' (Profile: '{profile}'). "
                "Please configure it via LLMConfig."
            )
        return key

    def _create_openai(self, model_name: str, profile: Optional[str], timeout: int) -> BaseChatModel:
        try:
            from langchain_openai import ChatOpenAI

            api_key = self._require_key("openai", profile)
            return ChatOpenAI(model=model_name, api_key=api_key, request_timeout=timeout)
        except ImportError:
            raise ImportError("Please install 'pico-agent[openai]' to use this provider.")

    def _create_azure(self, model_name: str, profile: Optional[str], timeout: int) -> BaseChatModel:
        try:
            import os

            from langchain_openai import AzureChatOpenAI

            api_key = self._require_key("azure", profile)
            return AzureChatOpenAI(
                azure_deployment=model_name,
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                api_key=api_key,
                request_timeout=timeout,
            )
        except ImportError:
            raise ImportError("Please install 'pico-agent[openai]' to use Azure OpenAI.")

    def _create_gemini(self, model_name: str, profile: Optional[str], timeout: int) -> BaseChatModel:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            api_key = self._require_key("google", profile)
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.0,
                request_timeout=timeout,
            )
        except ImportError:
            raise ImportError("Please install 'pico-agent[google]' to use Gemini.")

    def _create_anthropic(self, model_name: str, profile: Optional[str], timeout: int) -> BaseChatModel:
        try:
            from langchain_anthropic import ChatAnthropic

            api_key = self._require_key("anthropic", profile)
            base_url = self._get_base_url("anthropic", None, profile)
            return ChatAnthropic(
                model=model_name, api_key=api_key, base_url=base_url, temperature=0.0, default_request_timeout=timeout
            )
        except ImportError:
            raise ImportError("Please install 'pico-agent[anthropic]' to use Claude.")

    def _create_deepseek(self, model_name: str, profile: Optional[str], timeout: int) -> BaseChatModel:
        try:
            from langchain_openai import ChatOpenAI

            base_url = self._get_base_url("deepseek", "https://api.deepseek.com/v1", profile)
            api_key = self._require_key("deepseek", profile)
            return ChatOpenAI(
                model=model_name,
                base_url=base_url,
                api_key=api_key,
                temperature=0.0,
                request_timeout=timeout,
            )
        except ImportError:
            raise ImportError("Please install 'pico-agent[openai]' to use DeepSeek.")

    def _create_qwen(self, model_name: str, profile: Optional[str], timeout: int) -> BaseChatModel:
        try:
            from langchain_openai import ChatOpenAI

            base_url = self._get_base_url("qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1", profile)
            api_key = self._require_key("qwen", profile)
            return ChatOpenAI(
                model=model_name,
                base_url=base_url,
                api_key=api_key,
                temperature=0.0,
                request_timeout=timeout,
            )
        except ImportError:
            raise ImportError("Please install 'pico-agent[openai]' to use Qwen.")

    def create_chat_model(self, provider: str, model_name: str, profile: Optional[str]) -> BaseChatModel:
        """Create a LangChain ``BaseChatModel`` for the specified provider.

        Args:
            provider: Provider key (``"openai"``, ``"azure"``, ``"gemini"``,
                ``"google"``, ``"claude"``, ``"anthropic"``, ``"deepseek"``,
                ``"qwen"``).
            model_name: The provider-specific model name.
            profile: Optional named profile for API key / base URL.

        Returns:
            A configured ``BaseChatModel`` instance.

        Raises:
            ValueError: If *provider* is not recognised.
        """
        provider_lower = provider.lower()
        timeout = 60

        providers = {
            "openai": self._create_openai,
            "azure": self._create_azure,
            "gemini": self._create_gemini,
            "google": self._create_gemini,
            "claude": self._create_anthropic,
            "anthropic": self._create_anthropic,
            "deepseek": self._create_deepseek,
            "qwen": self._create_qwen,
        }

        creator = providers.get(provider_lower)
        if not creator:
            raise ValueError(f"Unknown LLM Provider: {provider}")

        return creator(model_name, profile, timeout)
