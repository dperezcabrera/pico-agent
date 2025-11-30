from typing import Optional, Any, Type
from pico_ioc import factory, provides, component, PicoContainer
from .interfaces import CentralConfigClient, LLMFactory
from .config import AgentConfig, LLMConfig
from .registry import AgentConfigService, ToolRegistry, LocalAgentRegistry
from .proxy import DynamicAgentProxy
from .virtual import VirtualAgentRunner
from .router import ModelRouter
from .providers import LangChainLLMFactory
from .experiments import ExperimentRegistry

class NoOpCentralClient(CentralConfigClient):
    def get_agent_config(self, name: str) -> Optional[AgentConfig]: return None
    def upsert_agent_config(self, config: AgentConfig) -> None: pass

@factory
class AgentInfrastructureFactory:
    def __init__(self, container: PicoContainer):
        self.container = container

    @provides(CentralConfigClient, scope="singleton")
    def provide_central_config(self) -> CentralConfigClient: 
        return NoOpCentralClient()
    
    @provides(LLMConfig, scope="singleton")
    def provide_llm_config(self) -> LLMConfig:
        return LLMConfig()

    @provides(LLMFactory, scope="singleton")
    def provide_llm_factory(self, config: LLMConfig) -> LLMFactory:
        return LangChainLLMFactory(config, self.container)

@component(scope="singleton")
class AgentLocator:
    def __init__(
        self,
        container: PicoContainer,
        config_service: AgentConfigService,
        tool_registry: ToolRegistry,
        llm_factory: LLMFactory,
        local_registry: LocalAgentRegistry,
        model_router: ModelRouter,
        experiment_registry: ExperimentRegistry
    ):
        self.container = container
        self.config_service = config_service
        self.tool_registry = tool_registry
        self.llm_factory = llm_factory
        self.local_registry = local_registry
        self.model_router = model_router
        self.experiment_registry = experiment_registry

    def get_agent(self, name_or_protocol: Any) -> Optional[Any]:
        agent_name = ""
        protocol = None

        if isinstance(name_or_protocol, type):
            protocol = name_or_protocol
            from .decorators import AGENT_META_KEY
            if hasattr(protocol, AGENT_META_KEY):
                agent_name = getattr(protocol, AGENT_META_KEY).name
            else:
                for n, p in self.local_registry._protocols.items():
                    if p == protocol:
                        agent_name = n
                        break
        else:
            requested_name = str(name_or_protocol)
            agent_name = self.experiment_registry.resolve_variant(requested_name)
            protocol = self.local_registry.get_protocol(agent_name)

        if not agent_name:
             return None

        if protocol:
            return self._create_proxy(agent_name, protocol)
        
        try:
            config = self.config_service.get_config(agent_name)
            if config:
                return VirtualAgentRunner(
                    config=config,
                    tool_registry=self.tool_registry,
                    llm_factory=self.llm_factory,
                    model_router=self.model_router,
                    container=self.container
                )
        except ValueError:
            pass

        return None

    def _create_proxy(self, name: str, protocol: Optional[Type]) -> Any:
        return DynamicAgentProxy(
            agent_name=name,
            protocol_cls=protocol,
            config_service=self.config_service,
            tool_registry=self.tool_registry,
            llm_factory=self.llm_factory,
            model_router=self.model_router,
            container=self.container,
            locator=self
        )
    
    def create_proxy(self, protocol: Type) -> Any:
        from .decorators import AGENT_META_KEY
        config = getattr(protocol, AGENT_META_KEY)
        return self._create_proxy(config.name, protocol)
