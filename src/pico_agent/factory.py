from typing import Optional, Any
from pico_ioc import factory, provides, component
from .interfaces import CentralConfigClient, LLMFactory
from .config import AgentConfig, LLMConfig
from .registry import AgentConfigService, ToolRegistry, LocalAgentRegistry
from .proxy import DynamicAgentProxy
from .router import ModelRouter
from .providers import LangChainLLMFactory

class NoOpCentralClient(CentralConfigClient):
    def get_agent_config(self, name: str) -> Optional[AgentConfig]: return None
    def upsert_agent_config(self, config: AgentConfig) -> None: pass

@factory
class AgentInfrastructureFactory:
    @provides(CentralConfigClient, scope="singleton")
    def provide_central_config(self) -> CentralConfigClient: 
        return NoOpCentralClient()
    
    @provides(LLMConfig, scope="singleton")
    def provide_llm_config(self) -> LLMConfig:
        return LLMConfig()

    @provides(LLMFactory, scope="singleton")
    def provide_llm_factory(self, config: LLMConfig) -> LLMFactory:
        return LangChainLLMFactory(config)

@component
class DynamicProxyFactory:
    def __init__(
        self,
        config_service: AgentConfigService,
        tool_registry: ToolRegistry,
        llm_factory: LLMFactory,
        local_registry: LocalAgentRegistry,
        model_router: ModelRouter
    ):
        self.config_service = config_service
        self.tool_registry = tool_registry
        self.llm_factory = llm_factory
        self.local_registry = local_registry
        self.model_router = model_router

    def get_agent(self, name: str) -> Optional[Any]:
        protocol = self.local_registry.get_protocol(name)
        if protocol:
            return self.create_proxy(protocol)
        return None

    def create_proxy(self, protocol: type) -> Any:
        from .decorators import AGENT_META_KEY
        config = getattr(protocol, AGENT_META_KEY)
        
        self.local_registry.register_default(config.name, config)
        self.local_registry.register_protocol(config.name, protocol)
        
        return DynamicAgentProxy(
            agent_name=config.name,
            protocol_cls=protocol,
            config_service=self.config_service,
            tool_registry=self.tool_registry,
            llm_factory=self.llm_factory,
            model_router=self.model_router,
            agent_resolver=self
        )
