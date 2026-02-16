"""Agent locator and infrastructure factory.

``AgentInfrastructureFactory`` provides default singletons for
``CentralConfigClient``, ``LLMConfig``, and ``LLMFactory``.
``AgentLocator`` is the primary entry point for obtaining agent proxies --
it resolves names (or Protocol classes) to ``DynamicAgentProxy`` or
``VirtualAgentRunner`` instances.
"""

from typing import Any, Optional, Type

from pico_ioc import PicoContainer, component, factory, provides

from .config import AgentConfig, LLMConfig
from .decorators import AGENT_META_KEY
from .experiments import ExperimentRegistry
from .interfaces import CentralConfigClient, LLMFactory
from .providers import LangChainLLMFactory
from .proxy import DynamicAgentProxy
from .registry import AgentConfigService, LocalAgentRegistry, ToolRegistry
from .router import ModelRouter
from .scheduler import PlatformScheduler
from .virtual import VirtualAgentRunner


class NoOpCentralClient(CentralConfigClient):
    """Default ``CentralConfigClient`` that returns ``None`` for all lookups.

    Used when no remote configuration backend is configured.
    """

    def get_agent_config(self, name: str) -> Optional[AgentConfig]:
        """Always returns ``None`` (no remote config)."""
        return None

    def upsert_agent_config(self, config: AgentConfig) -> None:
        """No-op -- remote persistence is not available."""
        pass


@factory
class AgentInfrastructureFactory:
    """Factory that registers default singletons for the agent infrastructure.

    Provides:

    - ``CentralConfigClient`` -- ``NoOpCentralClient`` (no remote config).
    - ``LLMConfig`` -- empty config; populate via ``@configure``.
    - ``LLMFactory`` -- ``LangChainLLMFactory`` wired to ``LLMConfig``.

    .. important::

        ``LLMConfig`` is registered here as a singleton.  To populate it with
        API keys, use ``@configure`` on a component method.  Do **not** create
        a competing ``@factory`` + ``@provides`` for ``LLMConfig`` -- that
        would conflict with this registration.

    Args:
        container: The pico-ioc container.
    """

    def __init__(self, container: PicoContainer):
        self.container = container

    @provides(CentralConfigClient, scope="singleton")
    def provide_central_config(self) -> CentralConfigClient:
        """Provide the default no-op central configuration client.

        Returns:
            A ``NoOpCentralClient`` instance.
        """
        return NoOpCentralClient()

    @provides(LLMConfig, scope="singleton")
    def provide_llm_config(self) -> LLMConfig:
        """Provide the default empty ``LLMConfig`` singleton.

        Populate it in a ``@configure`` hook, **not** with another
        ``@provides``.

        Returns:
            An empty ``LLMConfig`` instance.
        """
        return LLMConfig()

    @provides(LLMFactory, scope="singleton")
    def provide_llm_factory(self, config: LLMConfig) -> LLMFactory:
        """Provide the ``LangChainLLMFactory`` wired to ``LLMConfig``.

        Args:
            config: The ``LLMConfig`` singleton (injected automatically).

        Returns:
            A ``LangChainLLMFactory`` instance.
        """
        return LangChainLLMFactory(config, self.container)


@component(scope="singleton")
class AgentLocator:
    """Primary entry point for obtaining agent proxies.

    Resolves agent names (or Protocol classes) to ``DynamicAgentProxy``
    (for code-defined agents) or ``VirtualAgentRunner`` (for YAML-defined
    / runtime agents).  Supports A/B experiment resolution via
    ``ExperimentRegistry``.

    Args:
        container: The pico-ioc container.
        config_service: Service for resolving agent configurations.
        tool_registry: Registry for tool lookup.
        llm_factory: Factory for creating LLM instances.
        local_registry: Registry of locally discovered agents.
        model_router: Capability-to-model router.
        experiment_registry: A/B experiment variant selector.
        scheduler: Concurrency scheduler for async operations.
    """
    def __init__(
        self,
        container: PicoContainer,
        config_service: AgentConfigService,
        tool_registry: ToolRegistry,
        llm_factory: LLMFactory,
        local_registry: LocalAgentRegistry,
        model_router: ModelRouter,
        experiment_registry: ExperimentRegistry,
        scheduler: PlatformScheduler,
    ):
        self.container = container
        self.config_service = config_service
        self.tool_registry = tool_registry
        self.llm_factory = llm_factory
        self.local_registry = local_registry
        self.model_router = model_router
        self.experiment_registry = experiment_registry
        self.scheduler = scheduler

    def get_agent(self, name_or_protocol: Any) -> Optional[Any]:
        """Retrieve an agent proxy by name or Protocol class.

        Resolution order:

        1. If *name_or_protocol* is a type, look up the agent name from its
           ``AGENT_META_KEY`` metadata or from ``LocalAgentRegistry``.
        2. If a string, pass through ``ExperimentRegistry.resolve_variant()``
           for A/B testing support.
        3. If a matching Protocol exists locally, create a
           ``DynamicAgentProxy``.
        4. Otherwise, attempt to create a ``VirtualAgentRunner`` from config.

        Args:
            name_or_protocol: Either an agent name string or a Protocol class.

        Returns:
            A ``DynamicAgentProxy`` or ``VirtualAgentRunner``, or ``None`` if
            no agent could be resolved.
        """
        agent_name = ""
        protocol = None

        if isinstance(name_or_protocol, type):
            protocol = name_or_protocol
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
                    container=self.container,
                    locator=self,
                    scheduler=self.scheduler,
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
            locator=self,
        )

    def create_proxy(self, protocol: Type) -> Any:
        """Create a ``DynamicAgentProxy`` directly from a Protocol class.

        Convenience method that extracts the agent name from the Protocol's
        ``AGENT_META_KEY`` metadata.

        Args:
            protocol: A Protocol class decorated with ``@agent``.

        Returns:
            A ``DynamicAgentProxy`` for the given Protocol.
        """
        config = getattr(protocol, AGENT_META_KEY)
        return self._create_proxy(config.name, protocol)
