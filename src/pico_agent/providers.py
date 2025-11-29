import os
from typing import List, Dict, Any, Type, Optional
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from .interfaces import LLM, LLMFactory
from .config import LLMConfig

class LangChainAdapter(LLM):
    def __init__(self, chat_model: BaseChatModel):
        self.model = chat_model

    def _convert_messages(self, messages: List[Dict[str, str]]) -> List[BaseMessage]:
        lc_messages = []
        for msg in messages:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_messages.append(AIMessage(content=msg["content"]))
        return lc_messages

    def invoke(self, messages: List[Dict[str, str]], tools: List[Any]) -> str:
        lc_messages = self._convert_messages(messages)
        model_with_tools = self.model
        
        if tools:
            model_with_tools = self.model.bind_tools(tools)
        
        response = model_with_tools.invoke(lc_messages)
        return str(response.content)

    def invoke_structured(
        self, 
        messages: List[Dict[str, str]], 
        tools: List[Any], 
        output_schema: Type[Any]
    ) -> Any:
        lc_messages = self._convert_messages(messages)
        structured_model = self.model.with_structured_output(output_schema)
        return structured_model.invoke(lc_messages)

    def invoke_agent_loop(
        self, 
        messages: List[Dict[str, str]], 
        tools: List[Any], 
        max_iterations: int, 
        output_schema: Optional[Type[Any]] = None
    ) -> Any:
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

class LangChainLLMFactory(LLMFactory):
    def __init__(self, config: LLMConfig):
        self.config = config

    def create(self, model_name: str, temperature: float, max_tokens: Optional[int], llm_profile: Optional[str] = None) -> LLM:
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
                 
        return LangChainAdapter(chat_model)

    def _get_api_key(self, provider: str, profile: Optional[str]) -> Optional[str]:
        if profile and profile in self.config.api_keys:
            return self.config.api_keys[profile]
        return self.config.api_keys.get(provider)

    def _get_base_url(self, provider: str, default: Optional[str], profile: Optional[str]) -> Optional[str]:
        if profile and profile in self.config.base_urls:
            return self.config.base_urls[profile]
        return self.config.base_urls.get(provider, default)

    def _detect_provider(self, model_name: str) -> str:
        name_lower = model_name.lower()
        if "gemini" in name_lower: return "gemini"
        elif "claude" in name_lower or "anthropic" in name_lower: return "claude"
        elif "deepseek" in name_lower: return "deepseek"
        elif "qwen" in name_lower: return "qwen"
        elif "azure" in name_lower: return "azure"
        return "openai"

    def create_chat_model(self, provider: str, model_name: str, profile: Optional[str]) -> BaseChatModel:
        provider_lower = provider.lower()
        timeout = 60
        
        def require_key(p_name, key_val):
            if not key_val:
                raise ValueError(
                    f"API Key not found for provider '{p_name}' (Profile: '{profile}'). "
                    "Please configure it via LLMConfig."
                )
            return key_val
        
        if provider_lower == "openai":
            try:
                from langchain_openai import ChatOpenAI
                api_key = require_key("openai", self._get_api_key("openai", profile))
                return ChatOpenAI(model=model_name, api_key=api_key, request_timeout=timeout)
            except ImportError:
                raise ImportError("Please install 'pico-agent[openai]' to use this provider.")
                
        elif provider_lower == "azure":
            try:
                from langchain_openai import AzureChatOpenAI
                import os
                api_key = require_key("azure", self._get_api_key("azure", profile))
                return AzureChatOpenAI(
                    azure_deployment=model_name,
                    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"), 
                    api_key=api_key,
                    request_timeout=timeout,
                )
            except ImportError:
                raise ImportError("Please install 'pico-agent[openai]' to use Azure OpenAI.")

        elif provider_lower == "gemini" or provider_lower == "google":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                api_key = require_key("google", self._get_api_key("google", profile))
                return ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=api_key,
                    temperature=0.0,
                    request_timeout=timeout,
                )
            except ImportError:
                raise ImportError("Please install 'pico-agent[google]' to use Gemini.")

        elif provider_lower == "claude" or provider_lower == "anthropic":
            try:
                from langchain_anthropic import ChatAnthropic
                api_key = require_key("anthropic", self._get_api_key("anthropic", profile))
                base_url = self._get_base_url("anthropic", None, profile)
                return ChatAnthropic(
                    model=model_name, 
                    api_key=api_key,
                    base_url=base_url,
                    temperature=0.0, 
                    default_request_timeout=timeout
                )
            except ImportError:
                raise ImportError("Please install 'pico-agent[anthropic]' to use Claude.")

        elif provider_lower == "deepseek":
            try:
                from langchain_openai import ChatOpenAI
                base_url = self._get_base_url("deepseek", "https://api.deepseek.com/v1", profile)
                api_key = require_key("deepseek", self._get_api_key("deepseek", profile))
                return ChatOpenAI(
                    model=model_name,
                    base_url=base_url,
                    api_key=api_key,
                    temperature=0.0,
                    request_timeout=timeout,
                )
            except ImportError:
                raise ImportError("Please install 'pico-agent[openai]' to use DeepSeek.")

        elif provider_lower == "qwen":
            try:
                from langchain_openai import ChatOpenAI
                base_url = self._get_base_url("qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1", profile)
                api_key = require_key("qwen", self._get_api_key("qwen", profile))
                return ChatOpenAI(
                    model=model_name,
                    base_url=base_url,
                    api_key=api_key,
                    temperature=0.0,
                    request_timeout=timeout,
                )
            except ImportError:
                raise ImportError("Please install 'pico-agent[openai]' to use Qwen.")

        else:
            raise ValueError(f"Unknown LLM Provider: {provider}")
