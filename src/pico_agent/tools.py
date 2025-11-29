import inspect
from typing import Any, Type, get_type_hints
from pydantic import BaseModel, create_model

class AgentAsTool:
    def __init__(self, agent_proxy: Any, method_name: str = "invoke", description: str = ""):
        self.proxy = agent_proxy
        self.method_name = method_name
        self._func = getattr(agent_proxy, method_name)
        self.name = getattr(agent_proxy, "agent_name", "agent_tool")
        self.description = description or f"Agent {self.name}"
        self.args_schema = self._create_schema_from_sig()

    def _create_schema_from_sig(self) -> Type[BaseModel]:
        protocol_cls = self.proxy.protocol_cls
        real_method = getattr(protocol_cls, self.method_name)
        sig = inspect.signature(real_method)
        type_hints = get_type_hints(real_method)
        
        fields = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self": continue
            annotation = type_hints.get(param_name, str)
            default = param.default if param.default is not inspect.Parameter.empty else ...
            fields[param_name] = (annotation, default)
            
        return create_model(f"{self.name}Input", **fields)

    def __call__(self, **kwargs):
        return self._func(**kwargs)
