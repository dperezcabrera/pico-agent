"""Tool wrappers for pico-agent.

``ToolWrapper`` adapts pico-agent ``@tool``-decorated classes to the
LangChain tool interface.  ``AgentAsTool`` wraps child agents so they can
be invoked as tools by a parent agent during a ReAct loop.
"""

import inspect
from typing import Any, Optional, Type, get_type_hints

from pydantic import BaseModel, create_model

from .config import ToolConfig
from .logging import get_logger

logger = get_logger(__name__)


def _create_schema_from_sig(name: str, func_or_method: Any) -> Type[BaseModel]:
    """Build a Pydantic model from the signature of *func_or_method*.

    The generated model is used as the ``args_schema`` expected by LangChain
    tool invocation.

    Args:
        name: Base name for the generated model (suffixed with ``"Input"``).
        func_or_method: The callable whose parameters define the schema
            fields.

    Returns:
        A dynamically created ``pydantic.BaseModel`` subclass.
    """
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
    """Adapts a pico-agent ``@tool``-decorated instance to the LangChain tool interface.

    Exposes ``name``, ``description``, ``args_schema``, and ``__call__`` so
    the instance can be passed directly to LangChain tool-binding APIs.

    Args:
        instance: An instance of a ``@tool``-decorated class.
        config: The ``ToolConfig`` extracted from the class metadata.

    Raises:
        ValueError: If the instance does not implement ``__call__``, ``run``,
            ``execute``, or ``invoke``.  Message:
            ``"Tool <name> must implement __call__, run, execute, or invoke."``
    """

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
    """Wraps a ``DynamicAgentProxy`` as a LangChain-compatible tool.

    This allows a parent agent to invoke a child agent through the LLM's
    tool-calling mechanism.  The tool's ``args_schema`` is derived from the
    child agent's Protocol method signature.

    Args:
        agent_proxy: A ``DynamicAgentProxy`` for the child agent.
        method_name: The protocol method to invoke (default: ``"invoke"``).
        description: Optional description override.  If empty, the child
            agent's ``AgentConfig.description`` is used.
    """

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
