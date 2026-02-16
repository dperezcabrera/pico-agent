"""A/B experiment registry for agent variant selection.

``ExperimentRegistry`` allows mapping a public agent name to multiple
variant agent names with weighted random selection, enabling A/B testing
of different agent implementations.
"""

import random
from typing import Dict, List, Optional, Tuple

from pico_ioc import component


@component(scope="singleton")
class ExperimentRegistry:
    """Singleton registry for A/B experiments on agents.

    Register experiments mapping a public name to weighted variants.  When
    ``AgentLocator`` resolves an agent name, it passes through
    ``resolve_variant()`` to select the appropriate variant.

    Example:
        >>> registry = container.get(ExperimentRegistry)
        >>> registry.register_experiment("summarizer", {
        ...     "summarizer_v1": 0.8,
        ...     "summarizer_v2": 0.2,
        ... })
    """

    def __init__(self):
        self._experiments: Dict[str, List[Tuple[str, float]]] = {}

    def register_experiment(self, public_name: str, variants: Dict[str, float]):
        """Register an A/B experiment.

        Weights are normalised so they sum to 1.0.

        Args:
            public_name: The public agent name that triggers variant
                selection.
            variants: Mapping of variant agent names to their relative
                weights.
        """
        total_weight = sum(variants.values())
        normalized_variants = []

        for name, weight in variants.items():
            normalized_variants.append((name, weight / total_weight))

        self._experiments[public_name] = normalized_variants

    def resolve_variant(self, name: str) -> str:
        """Resolve a public name to a variant using weighted random selection.

        If no experiment is registered for *name*, the name itself is
        returned unchanged.

        Args:
            name: The public agent name.

        Returns:
            The selected variant name, or *name* if no experiment exists.
        """
        if name not in self._experiments:
            return name

        variants = self._experiments[name]
        choices = [v[0] for v in variants]
        weights = [v[1] for v in variants]

        return random.choices(choices, weights=weights, k=1)[0]
