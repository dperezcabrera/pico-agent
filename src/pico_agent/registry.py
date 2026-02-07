from dataclasses import replace
from typing import Any, Dict, List, Optional, Type

from pico_ioc import component

from .config import AgentConfig
from .interfaces import CentralConfigClient


@component
class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Any] = {}
        self._tag_map: Dict[str, List[str]] = {}

    def register(self, name: str, tool_cls_or_instance: Any, tags: Optional[List[str]] = None) -> None:
        tags = tags or []
        self._tools[name] = tool_cls_or_instance
        for tag in tags:
            if tag not in self._tag_map:
                self._tag_map[tag] = []
            self._tag_map[tag].append(name)

    def get_tool(self, name: str) -> Optional[Any]:
        return self._tools.get(name)

    def get_tool_names_by_tag(self, tag: str) -> List[str]:
        return self._tag_map.get(tag, [])

    def get_dynamic_tools(self, agent_tags: List[str]) -> List[Any]:
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
    def __init__(self):
        self._configs: Dict[str, AgentConfig] = {}
        self._protocols: Dict[str, Type] = {}

    def register(self, name: str, protocol: Type, config: AgentConfig) -> None:
        self._configs[name] = config
        self._protocols[name] = protocol

    def get_config(self, name: str) -> Optional[AgentConfig]:
        return self._configs.get(name)

    def get_protocol(self, name: str) -> Optional[Type]:
        return self._protocols.get(name)


@component
class AgentConfigService:
    def __init__(self, central_client: CentralConfigClient, local_registry: LocalAgentRegistry):
        self.central_client = central_client
        self.local_registry = local_registry
        self.auto_register = True
        self._runtime_overrides: Dict[str, Dict[str, Any]] = {}

    def get_config(self, name: str) -> AgentConfig:
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
        if name not in self._runtime_overrides:
            self._runtime_overrides[name] = {}
        self._runtime_overrides[name].update(kwargs)

    def reset_agent_config(self, name: str):
        if name in self._runtime_overrides:
            del self._runtime_overrides[name]
