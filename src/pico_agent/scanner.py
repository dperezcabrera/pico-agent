"""Auto-discovery scanners for agents and tools.

``AgentScanner`` and ``ToolScanner`` walk the call-stack modules during the
pico-ioc ``@configure`` phase to find classes decorated with ``@agent`` and
``@tool``, then register them in their respective registries.

Infrastructure modules (pico-ioc, pico-agent, importlib, pytest, etc.) are
skipped automatically.
"""

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
    """Shared auto-scan logic for ``AgentScanner`` and ``ToolScanner``.

    The ``auto_scan`` method is decorated with ``@configure`` so that pico-ioc
    invokes it during container initialisation.  It walks the Python call
    stack, inspects each module, and delegates to the subclass's
    ``scan_module`` method.
    """

    _scanned_modules: Set[str]

    @configure
    def auto_scan(self):
        """Walk the call stack and scan each non-infrastructure module.

        This method is called automatically by pico-ioc during the
        ``@configure`` phase.
        """
        frame = inspect.currentframe()
        while frame:
            mod = inspect.getmodule(frame)
            if mod and mod.__name__ and not _is_infrastructure(mod.__name__):
                self.scan_module(mod)
            frame = frame.f_back


@component
class AgentScanner(_ScannerBase):
    """Discovers ``@agent``-decorated Protocol classes and registers them.

    Walks Python modules to find classes that carry the ``IS_AGENT_INTERFACE``
    flag, extracts the ``AgentConfig`` from ``AGENT_META_KEY``, and stores
    both the Protocol class and its config in ``LocalAgentRegistry``.

    Args:
        registry: The ``LocalAgentRegistry`` to populate.
    """

    def __init__(self, registry: LocalAgentRegistry):
        self.registry = registry
        self._scanned_modules: Set[str] = set()

    def scan_module(self, module: Any):
        """Inspect a module for ``@agent``-decorated classes.

        Each discovered class is registered in the ``LocalAgentRegistry``.
        Modules are scanned at most once.

        Args:
            module: A Python module object.
        """
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
    """Discovers ``@tool``-decorated classes and registers them.

    Walks Python modules to find classes that carry ``TOOL_META_KEY``,
    extracts the ``ToolConfig``, and stores the class in ``ToolRegistry``.

    Args:
        registry: The ``ToolRegistry`` to populate.
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self._scanned_modules: Set[str] = set()

    def scan_module(self, module: Any):
        """Inspect a module for ``@tool``-decorated classes.

        Each discovered class is registered in the ``ToolRegistry``.
        Modules are scanned at most once.

        Args:
            module: A Python module object.
        """
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
