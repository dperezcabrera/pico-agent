import inspect
from typing import Set, Any
from pico_ioc import component, configure
from .decorators import IS_AGENT_INTERFACE, AGENT_META_KEY, TOOL_META_KEY
from .registry import LocalAgentRegistry, ToolRegistry

@component
class AgentScanner:
    def __init__(self, registry: LocalAgentRegistry):
        self.registry = registry
        self._scanned_modules: Set[str] = set()

    @configure
    def auto_scan(self):
        frame = inspect.currentframe()
        while frame:
            mod = inspect.getmodule(frame)
            if mod and mod.__name__ and not self._is_infrastructure(mod.__name__):
                self.scan_module(mod)
            frame = frame.f_back

    def _is_infrastructure(self, name: str) -> bool:
        return name.startswith(("pico_ioc", "pico_agent", "importlib", "contextlib", "pytest", "_pytest", "pluggy"))

    def scan_module(self, module: Any):
        mod_name = module.__name__
        if mod_name in self._scanned_modules:
            return
        self._scanned_modules.add(mod_name)

        try:
            members = inspect.getmembers(module)
        except Exception:
            return

        for name, obj in members:
            if isinstance(obj, type) and getattr(obj, IS_AGENT_INTERFACE, False):
                config = getattr(obj, AGENT_META_KEY)
                self.registry.register(config.name, obj, config)

@component
class ToolScanner:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self._scanned_modules: Set[str] = set()

    @configure
    def auto_scan(self):
        frame = inspect.currentframe()
        while frame:
            mod = inspect.getmodule(frame)
            if mod and mod.__name__ and not self._is_infrastructure(mod.__name__):
                self.scan_module(mod)
            frame = frame.f_back

    def _is_infrastructure(self, name: str) -> bool:
        return name.startswith(("pico_ioc", "pico_agent", "importlib", "contextlib", "pytest", "_pytest", "pluggy"))

    def scan_module(self, module: Any):
        mod_name = module.__name__
        if mod_name in self._scanned_modules:
            return
        self._scanned_modules.add(mod_name)

        try:
            members = inspect.getmembers(module)
        except Exception:
            return

        for name, obj in members:
            if isinstance(obj, type) and hasattr(obj, TOOL_META_KEY):
                config = getattr(obj, TOOL_META_KEY)
                self.registry.register(config.name, obj)
