"""Capability-to-model routing.

``ModelRouter`` translates abstract ``AgentCapability`` labels into concrete
provider-specific model names, enabling global model changes without
modifying individual agent definitions.
"""

from typing import Dict, Optional

from pico_ioc import component

from .config import AgentCapability


@component(scope="singleton")
class ModelRouter:
    """Maps ``AgentCapability`` labels to concrete LLM model names.

    Default mappings:

    ==============================  ========================
    Capability                      Default model
    ==============================  ========================
    ``AgentCapability.FAST``        ``gpt-5-mini``
    ``AgentCapability.SMART``       ``gpt-5.1``
    ``AgentCapability.REASONING``   ``gemini-3-pro``
    ``AgentCapability.VISION``      ``gpt-4o``
    ``AgentCapability.CODING``      ``claude-3-5-sonnet``
    ==============================  ========================

    Use ``update_mapping()`` to change a mapping at runtime.
    """

    def __init__(self):
        self._capability_map: Dict[str, str] = {
            AgentCapability.FAST: "gpt-5-mini",
            AgentCapability.SMART: "gpt-5.1",
            AgentCapability.REASONING: "gemini-3-pro",
            AgentCapability.VISION: "gpt-4o",
            AgentCapability.CODING: "claude-3-5-sonnet",
        }

    def resolve_model(self, capability: str, runtime_override: Optional[str] = None) -> str:
        """Resolve a capability label to a model name.

        If a *runtime_override* is provided it takes precedence over the
        capability mapping.

        Args:
            capability: An ``AgentCapability`` constant (e.g., ``"smart"``).
            runtime_override: Explicit model name that bypasses the mapping.

        Returns:
            The model name string.
        """
        if runtime_override:
            return runtime_override

        return self._capability_map.get(capability, "gpt-5.1")

    def update_mapping(self, capability: str, model: str) -> None:
        """Change the model associated with a capability.

        Args:
            capability: The ``AgentCapability`` constant to update.
            model: The new model name.
        """
        self._capability_map[capability] = model
