from typing import Any, Optional, Tuple, Callable
from pico_ioc.factory import DeferredProvider, ProviderMetadata
from .decorators import IS_AGENT_INTERFACE, AGENT_META_KEY

class AgentScanner:
    def should_scan(self, obj: Any) -> bool:
        return isinstance(obj, type) and getattr(obj, IS_AGENT_INTERFACE, False)

    def scan(self, obj: Any) -> Optional[Tuple[Any, Callable[[], Any], ProviderMetadata]]:
        if not self.should_scan(obj):
            return None

        config = getattr(obj, AGENT_META_KEY)

        def builder(container, locator):
            from .factory import DynamicProxyFactory
            factory = container.get(DynamicProxyFactory)
            return factory.create_proxy(obj)

        provider = DeferredProvider(builder)

        metadata = ProviderMetadata(
            key=obj,
            provided_type=obj,
            concrete_class=None,
            factory_class=None,
            factory_method=None,
            qualifiers=set(),
            primary=True,
            lazy=True,
            infra="pico_agent",
            pico_name=config.name,
            scope="singleton",
            dependencies=()
        )

        return (obj, provider, metadata)
