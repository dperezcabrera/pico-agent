from typing import Dict, Any, Optional, List
from pico_ioc import component
from .config import AgentConfig
from .interfaces import CentralConfigClient

@component
class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Any] = {}
        self._tag_map: Dict[str, List[str]] = {}

    def register(self, name: str, tool: Any, tags: List[str] = []) -> None:
        self._tools[name] = tool
        for tag in tags:
            if tag not in self._tag_map:
                self._tag_map[tag] = []
            self._tag_map[tag].append(name)

    def get_tool(self, name: str) -> Optional[Any]:
        return self._tools.get(name)

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
        self._defaults: Dict[str, AgentConfig] = {}
        self._protocols: Dict[str, type] = {}

    def register_default(self, name: str, config: AgentConfig) -> None:
        self._defaults[name] = config

    def register_protocol(self, name: str, protocol: type) -> None:
        self._protocols[name] = protocol

    def get_default(self, name: str) -> Optional[AgentConfig]:
        return self._defaults.get(name)

    def get_protocol(self, name: str) -> Optional[type]:
        return self._protocols.get(name)

@component
class AgentConfigService:
    def __init__(self, central_client: CentralConfigClient, local_registry: LocalAgentRegistry):
        self.central_client = central_client
        self.local_registry = local_registry
        self.auto_register = True

    def get_config(self, name: str) -> AgentConfig:
        remote_config = self.central_client.get_agent_config(name)
        if remote_config:
            return remote_config
        
        local_config = self.local_registry.get_default(name)
        if not local_config:
            raise ValueError(f"No configuration found for agent: {name}")

        if self.auto_register:
            try:
                self.central_client.upsert_agent_config(local_config)
            except Exception:
                pass
        return local_config
