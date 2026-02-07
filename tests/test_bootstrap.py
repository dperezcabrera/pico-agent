import os
import sys
import types

import pytest
from pico_ioc import PicoContainer

import pico_agent
from pico_agent.bootstrap import (
    _harvest_scanners,
    _import_module_like,
    _load_plugin_modules,
    _normalize_modules,
    _to_module_list,
    init,
)


class TestToModuleList:
    def test_list_passthrough(self):
        result = _to_module_list(["a", "b"])
        assert result == ["a", "b"]

    def test_single_item_wrapped(self):
        result = _to_module_list("mymodule")
        assert result == ["mymodule"]

    def test_tuple_passthrough(self):
        result = _to_module_list(("a", "b"))
        assert result == ["a", "b"]

    def test_module_object_wrapped(self):
        result = _to_module_list(pico_agent)
        assert result == [pico_agent]


class TestImportModuleLike:
    def test_module_object(self):
        result = _import_module_like(pico_agent)
        assert result is pico_agent

    def test_string_import(self):
        result = _import_module_like("pico_agent")
        assert result is pico_agent

    def test_invalid_object_raises(self):
        obj = object()
        with pytest.raises(ImportError):
            _import_module_like(obj)


class TestNormalizeModules:
    def test_deduplicates(self):
        result = _normalize_modules(["pico_agent", "pico_agent", pico_agent])
        names = [m.__name__ for m in result]
        assert names.count("pico_agent") == 1

    def test_preserves_order(self):
        result = _normalize_modules(["pico_agent", "os", "sys"])
        names = [m.__name__ for m in result]
        assert names == ["pico_agent", "os", "sys"]

    def test_mixed_types(self):
        result = _normalize_modules([pico_agent, "os"])
        assert len(result) == 2


class TestHarvestScanners:
    def test_no_scanners(self):
        mod = types.ModuleType("empty_mod")
        result = _harvest_scanners([mod])
        assert result == []

    def test_harvests_pico_scanners(self):
        mod = types.ModuleType("scanner_mod")
        mod.PICO_SCANNERS = ["ScannerA", "ScannerB"]
        result = _harvest_scanners([mod])
        assert result == ["ScannerA", "ScannerB"]

    def test_multiple_modules(self):
        mod1 = types.ModuleType("mod1")
        mod1.PICO_SCANNERS = ["S1"]
        mod2 = types.ModuleType("mod2")
        mod2.PICO_SCANNERS = ["S2"]
        result = _harvest_scanners([mod1, mod2])
        assert result == ["S1", "S2"]


class TestLoadPluginModules:
    def test_plugin_load_skips_infrastructure(self):
        modules = _load_plugin_modules(group="pico_agent.plugins")
        names = [m.__name__ for m in modules]
        assert "pico_ioc" not in names
        assert "pico_agent" not in names


class TestInit:
    def test_init_returns_container(self):
        container = init(modules=["pico_agent"])
        assert isinstance(container, PicoContainer)

    def test_pico_agent_always_included(self):
        from pico_agent.lifecycle import AgentSystem

        container = init(modules=["os"])
        assert container.has(AgentSystem)

    def test_auto_plugins_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("PICO_AGENT_AUTO_PLUGINS", "false")
        container = init(modules=["pico_agent"])
        assert isinstance(container, PicoContainer)

    def test_auto_plugins_disabled_via_zero(self, monkeypatch):
        monkeypatch.setenv("PICO_AGENT_AUTO_PLUGINS", "0")
        container = init(modules=["pico_agent"])
        assert isinstance(container, PicoContainer)

    def test_scanner_harvesting(self):
        mod = types.ModuleType("test_harvest_mod")
        mod.PICO_SCANNERS = []
        sys.modules["test_harvest_mod"] = mod
        try:
            container = init(modules=["pico_agent", "test_harvest_mod"])
            assert isinstance(container, PicoContainer)
        finally:
            del sys.modules["test_harvest_mod"]
