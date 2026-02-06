import pytest
from unittest.mock import MagicMock, Mock, patch
from types import ModuleType

from pico_agent.scanner import AgentScanner, ToolScanner
from pico_agent.registry import LocalAgentRegistry, ToolRegistry
from pico_agent.config import AgentConfig
from pico_agent.decorators import IS_AGENT_INTERFACE, AGENT_META_KEY, TOOL_META_KEY


class TestAgentScanner:
    @pytest.fixture
    def scanner(self, local_registry):
        return AgentScanner(local_registry)

    def test_is_infrastructure_pico_ioc(self, scanner):
        assert scanner._is_infrastructure("pico_ioc.container") is True

    def test_is_infrastructure_pico_agent(self, scanner):
        assert scanner._is_infrastructure("pico_agent.proxy") is True

    def test_is_infrastructure_pytest(self, scanner):
        assert scanner._is_infrastructure("pytest") is True
        assert scanner._is_infrastructure("_pytest.fixtures") is True

    def test_is_infrastructure_user_module(self, scanner):
        assert scanner._is_infrastructure("my_app.agents") is False
        assert scanner._is_infrastructure("business.logic") is False

    def test_scan_module_skips_already_scanned(self, scanner, local_registry):
        module = ModuleType("test_module_1")

        scanner.scan_module(module)
        scanner.scan_module(module)  # Second scan

        assert "test_module_1" in scanner._scanned_modules
        # Should only process once
        assert scanner._scanned_modules == {"test_module_1"}

    def test_scan_module_registers_agent_interface(self, scanner, local_registry):
        module = ModuleType("agents_module")

        @property
        def is_agent(self):
            return True

        # Create a mock agent class
        class MockAgent:
            pass

        config = AgentConfig(name="mock_agent", system_prompt="Test")
        setattr(MockAgent, IS_AGENT_INTERFACE, True)
        setattr(MockAgent, AGENT_META_KEY, config)

        module.MockAgent = MockAgent

        scanner.scan_module(module)

        assert local_registry.get_config("mock_agent") is not None
        assert local_registry.get_config("mock_agent").name == "mock_agent"

    def test_scan_module_handles_inspect_exception(self, scanner, local_registry):
        module = MagicMock()
        module.__name__ = "broken_module"

        with patch("inspect.getmembers", side_effect=TypeError("Cannot inspect")):
            # Should not raise, just return early
            scanner.scan_module(module)

        assert "broken_module" in scanner._scanned_modules

    def test_scan_module_handles_module_not_found(self, scanner):
        module = MagicMock()
        module.__name__ = "missing_deps_module"

        with patch("inspect.getmembers", side_effect=ModuleNotFoundError("No module")):
            scanner.scan_module(module)

        assert "missing_deps_module" in scanner._scanned_modules


class TestToolScanner:
    @pytest.fixture
    def scanner(self, tool_registry):
        return ToolScanner(tool_registry)

    def test_is_infrastructure_pico_ioc(self, scanner):
        assert scanner._is_infrastructure("pico_ioc.container") is True

    def test_is_infrastructure_pico_agent(self, scanner):
        assert scanner._is_infrastructure("pico_agent.tools") is True

    def test_is_infrastructure_user_module(self, scanner):
        assert scanner._is_infrastructure("my_app.tools") is False

    def test_scan_module_skips_already_scanned(self, scanner):
        module = ModuleType("tools_module")

        scanner.scan_module(module)
        scanner.scan_module(module)

        assert "tools_module" in scanner._scanned_modules

    def test_scan_module_registers_tool(self, scanner, tool_registry):
        from pico_agent.config import ToolConfig

        module = ModuleType("my_tools")

        class MyTool:
            pass

        tool_config = ToolConfig(name="my_tool", description="A test tool")
        setattr(MyTool, TOOL_META_KEY, tool_config)

        module.MyTool = MyTool

        scanner.scan_module(module)

        assert tool_registry.get_tool("my_tool") is not None

    def test_scan_module_handles_inspect_exception(self, scanner):
        module = MagicMock()
        module.__name__ = "broken_tool_module"

        with patch("inspect.getmembers", side_effect=TypeError("Cannot inspect")):
            scanner.scan_module(module)

        assert "broken_tool_module" in scanner._scanned_modules

    def test_scan_module_handles_module_not_found(self, scanner):
        module = MagicMock()
        module.__name__ = "missing_module"

        with patch("inspect.getmembers", side_effect=ModuleNotFoundError("No module")):
            scanner.scan_module(module)

        assert "missing_module" in scanner._scanned_modules


class TestScannerInfrastructureDetection:
    """Test that scanners correctly identify infrastructure modules."""

    @pytest.fixture
    def agent_scanner(self, local_registry):
        return AgentScanner(local_registry)

    @pytest.fixture
    def tool_scanner(self, tool_registry):
        return ToolScanner(tool_registry)

    @pytest.mark.parametrize("module_name,expected", [
        ("pico_ioc", True),
        ("pico_ioc.container", True),
        ("pico_agent", True),
        ("pico_agent.proxy", True),
        ("importlib", True),
        ("importlib.util", True),
        ("contextlib", True),
        ("pytest", True),
        ("_pytest", True),
        ("_pytest.fixtures", True),
        ("pluggy", True),
        ("pluggy.hooks", True),
        ("myapp", False),
        ("myapp.agents", False),
        ("business.services", False),
    ])
    def test_infrastructure_detection(self, agent_scanner, tool_scanner, module_name, expected):
        assert agent_scanner._is_infrastructure(module_name) == expected
        assert tool_scanner._is_infrastructure(module_name) == expected
