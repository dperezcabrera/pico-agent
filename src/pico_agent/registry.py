"""Registries for tools and agent configurations.

Provides ``ToolRegistry`` (tool storage with tag-based lookup),
``LocalAgentRegistry`` (stores configs discovered by ``AgentScanner``),
and ``AgentConfigService`` (merges central, local, and runtime config).
"""

from dataclasses import replace
from typing import Any, Dict, List, Optional, Type

from pico_ioc import component

from .config import AgentConfig
from .interfaces import CentralConfigClient


@component
class ToolRegistry:
    """Central registry that stores tool classes/instances and supports tag-based lookup.

    Tools are registered by ``ToolScanner`` during auto-discovery or manually
    via ``register()``.  At execution time, ``DynamicAgentProxy`` and
    ``VirtualAgentRunner`` query this registry to resolve tool dependencies.
    """

    def __init__(self):
        self._tools: Dict[str, Any] = {}
        self._tag_map: Dict[str, List[str]] = {}

    def register(self, name: str, tool_cls_or_instance: Any, tags: Optional[List[str]] = None) -> None:
        """Register a tool by name with optional tags.

        Args:
            name: Unique tool identifier.
            tool_cls_or_instance: The tool class or an already-instantiated
                tool object.
            tags: Optional list of tags for dynamic tool lookup.  Tools
                tagged ``"global"`` are attached to every agent automatically.
        """
        tags = tags or []
        self._tools[name] = tool_cls_or_instance
        for tag in tags:
            if tag not in self._tag_map:
                self._tag_map[tag] = []
            self._tag_map[tag].append(name)

    def get_tool(self, name: str) -> Optional[Any]:
        """Retrieve a tool by name.

        Args:
            name: The tool identifier.

        Returns:
            The tool class or instance, or ``None`` if not found.
        """
        return self._tools.get(name)

    def get_tool_names_by_tag(self, tag: str) -> List[str]:
        """Return all tool names associated with the given tag.

        Args:
            tag: The tag to search for.

        Returns:
            List of matching tool names (may be empty).
        """
        return self._tag_map.get(tag, [])

    def get_dynamic_tools(self, agent_tags: List[str]) -> List[Any]:
        """Collect tool instances matching any of the given tags, plus ``"global"`` tools.

        Duplicates are excluded.

        Args:
            agent_tags: Tags from the agent's ``AgentConfig.tags``.

        Returns:
            De-duplicated list of tool instances.
        """
        found_tools = []
        for tag in agent_tags:
            tool_names = self._tag_map.get(tag, [])
            for name in tool_names:
                t = self._tools.get(name)
                if t and t not in found_tools:
                    found_tools.append(t)

        global_names = self._tag_map.get("global", [])
        for name in global_names:
            t = self._tools.get(name)
            if t and t not in found_tools:
                found_tools.append(t)
        return found_tools


@component
class LocalAgentRegistry:
    """In-memory store of agent Protocol classes and their ``AgentConfig`` metadata.

    Populated by ``AgentScanner`` during auto-discovery.
    """

    def __init__(self):
        self._configs: Dict[str, AgentConfig] = {}
        self._protocols: Dict[str, Type] = {}

    def register(self, name: str, protocol: Type, config: AgentConfig) -> None:
        """Register an agent protocol and its configuration.

        Args:
            name: Unique agent identifier.
            protocol: The Protocol class decorated with ``@agent``.
            config: The ``AgentConfig`` extracted from the decorator.
        """
        self._configs[name] = config
        self._protocols[name] = protocol

    def get_config(self, name: str) -> Optional[AgentConfig]:
        """Retrieve the locally registered ``AgentConfig``.

        Args:
            name: Agent identifier.

        Returns:
            The ``AgentConfig``, or ``None`` if not registered.
        """
        return self._configs.get(name)

    def get_protocol(self, name: str) -> Optional[Type]:
        """Retrieve the Protocol class for an agent.

        Args:
            name: Agent identifier.

        Returns:
            The Protocol class, or ``None`` if not registered.
        """
        return self._protocols.get(name)


@component
class AgentConfigService:
    """Merges central, local, and runtime configuration for agents.

    Configuration priority (highest wins): **central > local > runtime**.
    Central config (from ``CentralConfigClient``) takes precedence over the
    local config discovered by ``AgentScanner``.  Runtime overrides set via
    ``update_agent_config()`` are applied on top of whichever base config is
    found.

    Args:
        central_client: Remote configuration client.
        local_registry: Registry populated by ``AgentScanner``.
    """

    def __init__(self, central_client: CentralConfigClient, local_registry: LocalAgentRegistry):
        self.central_client = central_client
        self.local_registry = local_registry
        self.auto_register = True
        self._runtime_overrides: Dict[str, Dict[str, Any]] = {}

    def get_config(self, name: str) -> AgentConfig:
        """Return the effective ``AgentConfig`` for the named agent.

        Merges remote, local, and runtime sources.

        Args:
            name: Agent identifier.

        Returns:
            The merged ``AgentConfig``.

        Raises:
            ValueError: If no configuration exists for the given name.
                Message: ``"No configuration found for agent: <name>"``.
        """
        remote_config = self.central_client.get_agent_config(name)
        local_config = self.local_registry.get_config(name)

        base_config = remote_config or local_config
        runtime_data = self._runtime_overrides.get(name)

        if base_config:
            if runtime_data:
                return replace(base_config, **runtime_data)
            return base_config

        elif runtime_data:
            config_data = runtime_data.copy()
            if "name" not in config_data:
                config_data["name"] = name
            return AgentConfig(**config_data)

        raise ValueError(f"No configuration found for agent: {name}")

    def update_agent_config(self, name: str, **kwargs):
        """Apply runtime overrides to an agent's configuration.

        Overrides are merged on each ``get_config()`` call.

        Args:
            name: Agent identifier.
            **kwargs: Fields of ``AgentConfig`` to override.
        """
        if name not in self._runtime_overrides:
            self._runtime_overrides[name] = {}
        self._runtime_overrides[name].update(kwargs)

    def reset_agent_config(self, name: str):
        """Remove all runtime overrides for an agent.

        Args:
            name: Agent identifier.
        """
        if name in self._runtime_overrides:
            del self._runtime_overrides[name]
