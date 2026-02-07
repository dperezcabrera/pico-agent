from typing import Dict, Optional

from pico_ioc import component

from .config import AgentCapability


@component(scope="singleton")
class ModelRouter:
    def __init__(self):
        self._capability_map: Dict[str, str] = {
            AgentCapability.FAST: "gpt-5-mini",
            AgentCapability.SMART: "gpt-5.1",
            AgentCapability.REASONING: "gemini-3-pro",
            AgentCapability.VISION: "gpt-4o",
            AgentCapability.CODING: "claude-3-5-sonnet",
        }

    def resolve_model(self, capability: str, runtime_override: Optional[str] = None) -> str:
        if runtime_override:
            return runtime_override

        return self._capability_map.get(capability, "gpt-5.1")

    def update_mapping(self, capability: str, model: str) -> None:
        self._capability_map[capability] = model
