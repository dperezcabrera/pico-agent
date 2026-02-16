import inspect
from typing import Any, Optional, Type, get_type_hints

from pydantic import BaseModel, create_model

from .config import ToolConfig
from .logging import get_logger

logger = get_logger(__name__)


def _create_schema_from_sig(name: str, func_or_method: Any) -> Type[BaseModel]:
    """Build a Pydantic model from the signature of *func_or_method*."""
    sig = inspect.signature(func_or_method)
    type_hints = get_type_hints(func_or_method)

    fields = {}
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        annotation = type_hints.get(param_name, str)
        default = param.default if param.default is not inspect.Parameter.empty else ...
        fields[param_name] = (annotation, default)

    return create_model(f"{name}Input", **fields)


class ToolWrapper:
    def __init__(self, instance: Any, config: ToolConfig):
        self.instance = instance
        self.name = config.name
        self.description = config.description
        self.func = self._resolve_function(instance)
        self.args_schema = _create_schema_from_sig(self.name, self.func)

    def _resolve_function(self, instance: Any) -> Any:
        if hasattr(instance, "__call__"):
            return instance.__call__

        for method in ["run", "execute", "invoke"]:
            if hasattr(instance, method):
                return getattr(instance, method)

        raise ValueError(f"Tool {self.name} must implement __call__, run, execute, or invoke.")

    def __call__(self, **kwargs):
        return self.func(**kwargs)


class AgentAsTool:
    def __init__(self, agent_proxy: Any, method_name: str = "invoke", description: str = ""):
        self.proxy = agent_proxy
        self.method_name = method_name
        self._func = getattr(agent_proxy, method_name)
        self.name = getattr(agent_proxy, "agent_name", "agent_tool")

        if description:
            self.description = description
        else:
            config_service = getattr(agent_proxy, "config_service", None)
            if config_service:
                try:
                    cfg = config_service.get_config(self.name)
                    self.description = cfg.description or f"Agent {self.name}"
                except (ValueError, KeyError) as e:
                    logger.debug("Could not get config for agent %s: %s", self.name, e)
                    self.description = f"Agent {self.name}"
            else:
                self.description = f"Agent {self.name}"

        protocol_cls = self.proxy.protocol_cls
        real_method = getattr(protocol_cls, self.method_name)
        self.args_schema = _create_schema_from_sig(self.name, real_method)

    def __call__(self, **kwargs):
        return self._func(**kwargs)
