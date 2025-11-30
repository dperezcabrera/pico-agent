from typing import Any, Callable, Dict, List, Type, Optional
from pydantic import BaseModel, Field, create_model
from pico_ioc import component
from .registry import ToolRegistry
from .decorators import TOOL_META_KEY
from .config import ToolConfig

class DynamicTool:
    def __init__(
        self, 
        name: str, 
        description: str, 
        func: Callable[..., str], 
        args_schema: Type[BaseModel] = None
    ):
        self.name = name
        self.description = description
        self.func = func
        self.args_schema = args_schema or self._create_default_schema()
        
        config = ToolConfig(name=name, description=description)
        setattr(self, TOOL_META_KEY, config)

    def _create_default_schema(self) -> Type[BaseModel]:
        return create_model(
            f"{self.name}Input",
            payload=(List[Dict[str, Any]], Field(description="List of data dictionaries to process"))
        )

    def __call__(self, **kwargs):
        return self.func(**kwargs)

@component
class VirtualToolManager:
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry

    def create_tool(
        self, 
        name: str, 
        description: str, 
        func: Callable, 
        schema: Optional[Type[BaseModel]] = None
    ) -> DynamicTool:
        
        tool_instance = DynamicTool(
            name=name,
            description=description,
            func=func,
            args_schema=schema
        )
        
        self.tool_registry.register(name, tool_instance)
        
        return tool_instance

    def create_proto_tool(self, name: str, description: str, handler: Callable[[List[Dict[str, Any]]], str]):
        def wrapper(payload: List[Dict[str, Any]]) -> str:
            return handler(payload)
            
        return self.create_tool(name, description, wrapper)
