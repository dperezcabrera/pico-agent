import inspect
import os
from importlib import import_module
from importlib.metadata import entry_points
from types import ModuleType
from typing import TYPE_CHECKING, Any, Iterable, List, Union

from pico_ioc import init as _ioc_init

if TYPE_CHECKING:
    from pico_ioc import PicoContainer

import pico_agent

from .logging import get_logger

logger = get_logger(__name__)

_IOC_INIT_SIG = inspect.signature(_ioc_init)


def _to_module_list(modules: Union[Any, Iterable[Any]]) -> List[Any]:
    if isinstance(modules, Iterable) and not isinstance(modules, (str, bytes)):
        return list(modules)
    return [modules]


def _import_module_like(obj: Any) -> ModuleType:
    if isinstance(obj, ModuleType):
        return obj
    if isinstance(obj, str):
        return import_module(obj)
    module_name = getattr(obj, "__module__", None) or getattr(obj, "__name__", None)
    if not module_name:
        raise ImportError(f"Cannot determine module for object {obj!r}")
    return import_module(module_name)


def _normalize_modules(raw: Iterable[Any]) -> List[ModuleType]:
    seen: set[str] = set()
    result: List[ModuleType] = []
    for item in raw:
        m = _import_module_like(item)
        name = m.__name__
        if name not in seen:
            seen.add(name)
            result.append(m)
    return result


def _harvest_scanners(modules: List[ModuleType]) -> list:
    scanners: list = []
    for m in modules:
        module_scanners = getattr(m, "PICO_SCANNERS", None)
        if module_scanners:
            scanners.extend(module_scanners)
    return scanners


def _load_plugin_modules(group: str = "pico_agent.plugins") -> List[ModuleType]:
    selected = entry_points().select(group=group)

    seen: set[str] = set()
    modules: List[ModuleType] = []

    for ep in selected:
        try:
            if ep.module in ("pico_ioc", "pico_agent"):
                continue
            m = import_module(ep.module)
        except Exception as exc:
            logger.warning(
                "Failed to load pico-agent plugin entry point '%s' (%s): %s",
                ep.name,
                ep.module,
                exc,
            )
            continue

        name = m.__name__
        if name not in seen:
            seen.add(name)
            modules.append(m)

    return modules


def init(*args: Any, **kwargs: Any) -> "PicoContainer":
    bound = _IOC_INIT_SIG.bind(*args, **kwargs)
    bound.apply_defaults()

    raw = _to_module_list(bound.arguments["modules"])
    raw_with_agent = [pico_agent] + list(raw)
    base_modules = _normalize_modules(raw_with_agent)

    auto_flag = os.getenv("PICO_AGENT_AUTO_PLUGINS", "true").lower()
    if auto_flag not in ("0", "false", "no"):
        plugin_modules = _load_plugin_modules()
        all_modules = _normalize_modules(list(base_modules) + plugin_modules)
    else:
        all_modules = base_modules

    bound.arguments["modules"] = all_modules

    harvested = _harvest_scanners(all_modules)
    if harvested:
        existing = bound.arguments.get("custom_scanners") or []
        bound.arguments["custom_scanners"] = list(existing) + harvested

    return _ioc_init(*bound.args, **bound.kwargs)


init.__signature__ = _IOC_INIT_SIG
