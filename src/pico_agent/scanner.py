import inspect
from typing import Any, Set

from pico_ioc import component, configure
from pico_ioc.factory import DeferredProvider, ProviderMetadata

from .decorators import AGENT_META_KEY, IS_AGENT_INTERFACE, TOOL_META_KEY
from .logging import get_logger
from .registry import LocalAgentRegistry, ToolRegistry

logger = get_logger(__name__)

_INFRA_PREFIXES = ("pico_ioc", "pico_agent", "importlib", "contextlib", "pytest", "_pytest", "pluggy")


def _is_infrastructure(name: str) -> bool:
    """Return True if *name* belongs to an infrastructure module that scanners should skip."""
    return name.startswith(_INFRA_PREFIXES)


class _ScannerBase:
    """Shared auto-scan logic for AgentScanner and ToolScanner."""

    _scanned_modules: Set[str]

    @configure
    def auto_scan(self):
        frame = inspect.currentframe()
        while frame:
            mod = inspect.getmodule(frame)
            if mod and mod.__name__ and not _is_infrastructure(mod.__name__):
                self.scan_module(mod)
            frame = frame.f_back


@component
class AgentScanner(_ScannerBase):
    def __init__(self, registry: LocalAgentRegistry):
        self.registry = registry
        self._scanned_modules: Set[str] = set()

    def scan_module(self, module: Any):
        mod_name = module.__name__
        if mod_name in self._scanned_modules:
            return
        self._scanned_modules.add(mod_name)

        try:
            members = inspect.getmembers(module)
        except (TypeError, ModuleNotFoundError) as e:
            logger.warning("Cannot inspect module %s: %s", mod_name, e)
            return

        for name, obj in members:
            if isinstance(obj, type) and getattr(obj, IS_AGENT_INTERFACE, False):
                config = getattr(obj, AGENT_META_KEY)
                self.registry.register(config.name, obj, config)


@component
class ToolScanner(_ScannerBase):
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self._scanned_modules: Set[str] = set()

    def scan_module(self, module: Any):
        mod_name = module.__name__
        if mod_name in self._scanned_modules:
            return
        self._scanned_modules.add(mod_name)

        try:
            members = inspect.getmembers(module)
        except (TypeError, ModuleNotFoundError) as e:
            logger.warning("Cannot inspect module %s: %s", mod_name, e)
            return

        for name, obj in members:
            if isinstance(obj, type) and hasattr(obj, TOOL_META_KEY):
                config = getattr(obj, TOOL_META_KEY)
                self.registry.register(config.name, obj)
