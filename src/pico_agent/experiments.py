import random
from typing import Dict, List, Optional, Tuple

from pico_ioc import component


@component(scope="singleton")
class ExperimentRegistry:
    def __init__(self):
        self._experiments: Dict[str, List[Tuple[str, float]]] = {}

    def register_experiment(self, public_name: str, variants: Dict[str, float]):
        total_weight = sum(variants.values())
        normalized_variants = []

        for name, weight in variants.items():
            normalized_variants.append((name, weight / total_weight))

        self._experiments[public_name] = normalized_variants

    def resolve_variant(self, name: str) -> str:
        if name not in self._experiments:
            return name

        variants = self._experiments[name]
        choices = [v[0] for v in variants]
        weights = [v[1] for v in variants]

        return random.choices(choices, weights=weights, k=1)[0]
