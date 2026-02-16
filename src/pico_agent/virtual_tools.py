"""Runtime-created tools and the virtual tool manager.

``DynamicTool`` wraps a plain callable as a LangChain-compatible tool.
``VirtualToolManager`` creates ``DynamicTool`` instances and registers
them in the ``ToolRegistry``.
"""

from typing import Any, Callable, Dict, List, Optional, Type

from pico_ioc import component
from pydantic import BaseModel, Field, create_model

from .config import ToolConfig
from .decorators import TOOL_META_KEY
from .registry import ToolRegistry


class DynamicTool:
    """A tool created at runtime from a plain callable.

    Exposes ``name``, ``description``, ``args_schema``, and ``__call__``
    to satisfy the LangChain tool interface.

    Args:
        name: Unique tool identifier.
        description: Human-readable description for the LLM.
        func: The callable that implements the tool logic.
        args_schema: Optional Pydantic model for the tool's arguments.
            If ``None``, a default schema with a single ``payload`` field
            is generated.
    """

    def __init__(self, name: str, description: str, func: Callable[..., str], args_schema: Type[BaseModel] = None):
        self.name = name
        self.description = description
        self.func = func
        self.args_schema = args_schema or self._create_default_schema()

        config = ToolConfig(name=name, description=description)
        setattr(self, TOOL_META_KEY, config)

    def _create_default_schema(self) -> Type[BaseModel]:
        return create_model(
            f"{self.name}Input",
            payload=(List[Dict[str, Any]], Field(description="List of data dictionaries to process")),
        )

    def __call__(self, **kwargs):
        return self.func(**kwargs)


@component
class VirtualToolManager:
    """Creates and registers ``DynamicTool`` instances at runtime.

    Args:
        tool_registry: The ``ToolRegistry`` where created tools are stored.

    Example:
        >>> manager = container.get(VirtualToolManager)
        >>> tool = manager.create_tool(
        ...     name="echo",
        ...     description="Echoes the input back",
        ...     func=lambda text: text,
        ... )
    """

    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry

    def create_tool(
        self, name: str, description: str, func: Callable, schema: Optional[Type[BaseModel]] = None
    ) -> DynamicTool:
        """Create a ``DynamicTool`` and register it in the ``ToolRegistry``.

        Args:
            name: Unique tool identifier.
            description: Human-readable description for the LLM.
            func: The callable that implements the tool logic.
            schema: Optional Pydantic model for the tool's arguments.

        Returns:
            The created ``DynamicTool`` instance.
        """

        tool_instance = DynamicTool(name=name, description=description, func=func, args_schema=schema)

        self.tool_registry.register(name, tool_instance)

        return tool_instance

    def create_proto_tool(self, name: str, description: str, handler: Callable[[List[Dict[str, Any]]], str]):
        """Create a tool that accepts a list of dictionaries as its payload.

        Convenience wrapper around ``create_tool()`` for handlers that process
        structured batch data.

        Args:
            name: Unique tool identifier.
            description: Human-readable description for the LLM.
            handler: A callable that receives ``List[Dict[str, Any]]`` and
                returns a string result.

        Returns:
            The created ``DynamicTool`` instance.
        """
        def wrapper(payload: List[Dict[str, Any]]) -> str:
            return handler(payload)

        return self.create_tool(name, description, wrapper)
