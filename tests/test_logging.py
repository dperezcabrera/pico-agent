import io
import logging

import pytest

from pico_agent.logging import DEFAULT_FORMAT, configure_logging, get_logger


class TestGetLogger:
    def test_returns_logger(self):
        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)

    def test_adds_pico_agent_prefix(self):
        logger = get_logger("mymodule")
        assert logger.name == "pico_agent.mymodule"

    def test_preserves_existing_prefix(self):
        logger = get_logger("pico_agent.existing")
        assert logger.name == "pico_agent.existing"

    def test_handles_module_name(self):
        logger = get_logger(__name__)
        assert logger.name.startswith("pico_agent.")


class TestConfigureLogging:
    def teardown_method(self):
        # Clean up handlers after each test
        root_logger = logging.getLogger("pico_agent")
        root_logger.handlers.clear()

    def test_sets_log_level(self):
        configure_logging(level=logging.DEBUG)
        root_logger = logging.getLogger("pico_agent")
        assert root_logger.level == logging.DEBUG

    def test_adds_handler(self):
        configure_logging()
        root_logger = logging.getLogger("pico_agent")
        assert len(root_logger.handlers) == 1

    def test_does_not_duplicate_handlers(self):
        configure_logging()
        configure_logging()
        configure_logging()
        root_logger = logging.getLogger("pico_agent")
        assert len(root_logger.handlers) == 1

    def test_uses_custom_handler(self):
        stream = io.StringIO()
        custom_handler = logging.StreamHandler(stream)
        configure_logging(handler=custom_handler)

        logger = get_logger("test")
        logger.info("test message")

        output = stream.getvalue()
        assert "test message" in output

    def test_default_format_applied(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        configure_logging(level=logging.INFO, handler=handler)

        logger = get_logger("test")
        logger.info("formatted message")

        output = stream.getvalue()
        # Check format components
        assert "INFO" in output
        assert "pico_agent" in output
        assert "formatted message" in output


class TestLoggingIntegration:
    def teardown_method(self):
        root_logger = logging.getLogger("pico_agent")
        root_logger.handlers.clear()

    def test_child_loggers_inherit_level(self):
        configure_logging(level=logging.WARNING)

        logger = get_logger("child.module")
        assert logger.getEffectiveLevel() == logging.WARNING

    def test_multiple_loggers_same_namespace(self):
        configure_logging()

        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        # Both should share the same root handler
        root_logger = logging.getLogger("pico_agent")
        assert len(root_logger.handlers) == 1

    def test_log_output_includes_module_name(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        configure_logging(level=logging.INFO, handler=handler)

        logger = get_logger("specific_module")
        logger.info("module test")

        output = stream.getvalue()
        assert "specific_module" in output


class TestDefaultFormat:
    def test_format_string_components(self):
        assert "%(asctime)s" in DEFAULT_FORMAT
        assert "%(levelname)" in DEFAULT_FORMAT
        assert "%(name)s" in DEFAULT_FORMAT
        assert "%(message)s" in DEFAULT_FORMAT
