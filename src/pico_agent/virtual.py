from typing import Any, Protocol, List, Dict
from pico_ioc import component, PicoContainer
from .config import AgentConfig, AgentType
from .registry import AgentConfigService, ToolRegistry
from .interfaces import LLMFactory
from .router import ModelRouter
from .tools import ToolWrapper
from .decorators import TOOL_META_KEY

class VirtualAgent(Protocol):
    def run(self, input: str) -> str: ...

class VirtualAgentRunner:
    def __init__(
        self,
        config: AgentConfig,
        tool_registry: ToolRegistry,
        llm_factory: LLMFactory,
        model_router: ModelRouter,
        container: PicoContainer
    ):
        self.config = config
        self.tool_registry = tool_registry
        self.llm_factory = llm_factory
        self.model_router = model_router
        self.container = container

    def run(self, input: str) -> str:
        if not self.config.enabled:
            return "Agent is disabled."

        final_model_name = self.model_router.resolve_model(
            capability=self.config.capability,
            runtime_override=None
        )

        llm = self.llm_factory.create(
            model_name=final_model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            llm_profile=self.config.llm_profile
        )

        resolved_tools = self._resolve_tools()
        messages = self._build_messages(input)

        if self.config.agent_type == AgentType.REACT:
            return llm.invoke_agent_loop(
                messages,
                resolved_tools,
                self.config.max_iterations
            )
        else:
            return llm.invoke(messages, resolved_tools)

    def _resolve_tools(self) -> List[Any]:
        final_tools = []
        for tool_name in self.config.tools:
            tool_instance = None
            
            if self.container.has(tool_name):
                tool_instance = self.container.get(tool_name)
            
            elif self.tool_registry.get_tool(tool_name):
                tool_ref = self.tool_registry.get_tool(tool_name)
                if isinstance(tool_ref, type):
                    tool_instance = tool_ref()
                else:
                    tool_instance = tool_ref
            
            if tool_instance:
                if hasattr(tool_instance, "args_schema") and hasattr(tool_instance, "name"):
                    final_tools.append(tool_instance)
                elif hasattr(type(tool_instance), TOOL_META_KEY):
                    tool_config = getattr(type(tool_instance), TOOL_META_KEY)
                    final_tools.append(ToolWrapper(tool_instance, tool_config))
                else:
                    final_tools.append(tool_instance)
        
        return final_tools

    def _build_messages(self, user_input: str) -> List[Dict[str, str]]:
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        
        user_content = self.config.user_prompt_template.replace("{input}", user_input)
        messages.append({"role": "user", "content": user_content})
        return messages

@component
class VirtualAgentManager:
    def __init__(
        self,
        config_service: AgentConfigService,
        tool_registry: ToolRegistry,
        llm_factory: LLMFactory,
        model_router: ModelRouter,
        container: PicoContainer
    ):
        self.config_service = config_service
        self.tool_registry = tool_registry
        self.llm_factory = llm_factory
        self.model_router = model_router
        self.container = container

    def create_agent(self, name: str, **kwargs) -> VirtualAgent:
        config = AgentConfig(name=name, **kwargs)
        
        config_data = config.__dict__.copy()
        if "name" in config_data:
            del config_data["name"]

        self.config_service.update_agent_config(name, **config_data)
        
        return self.get_agent(name)

    def get_agent(self, name: str) -> VirtualAgent:
        config = self.config_service.get_config(name)
        return VirtualAgentRunner(
            config=config,
            tool_registry=self.tool_registry,
            llm_factory=self.llm_factory,
            model_router=self.model_router,
            container=self.container
        )
