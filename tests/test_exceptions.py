import pytest
from pico_agent.exceptions import AgentError, AgentDisabledError, AgentConfigurationError


class TestAgentError:
    def test_base_error_creation(self):
        error = AgentError("Something went wrong")
        assert str(error) == "Something went wrong"

    def test_is_exception(self):
        assert issubclass(AgentError, Exception)

    def test_can_be_raised(self):
        with pytest.raises(AgentError) as exc_info:
            raise AgentError("Test error")
        assert "Test error" in str(exc_info.value)


class TestAgentDisabledError:
    def test_error_message_format(self):
        error = AgentDisabledError("my_agent")
        assert "my_agent" in str(error)
        assert "disabled" in str(error).lower()

    def test_inherits_from_agent_error(self):
        assert issubclass(AgentDisabledError, AgentError)

    def test_can_catch_as_agent_error(self):
        with pytest.raises(AgentError):
            raise AgentDisabledError("test_agent")

    def test_specific_catch(self):
        with pytest.raises(AgentDisabledError) as exc_info:
            raise AgentDisabledError("disabled_agent")
        assert "disabled_agent" in str(exc_info.value)


class TestAgentConfigurationError:
    def test_error_creation(self):
        error = AgentConfigurationError("Missing API key")
        assert str(error) == "Missing API key"

    def test_inherits_from_agent_error(self):
        assert issubclass(AgentConfigurationError, AgentError)

    def test_can_catch_as_agent_error(self):
        with pytest.raises(AgentError):
            raise AgentConfigurationError("Config error")

    def test_specific_catch(self):
        with pytest.raises(AgentConfigurationError) as exc_info:
            raise AgentConfigurationError("API Key not found")
        assert "API Key not found" in str(exc_info.value)


class TestExceptionHierarchy:
    def test_all_errors_inherit_from_agent_error(self):
        errors = [AgentDisabledError, AgentConfigurationError]
        for error_cls in errors:
            assert issubclass(error_cls, AgentError)

    def test_agent_error_is_base_exception(self):
        assert issubclass(AgentError, Exception)
        assert not issubclass(Exception, AgentError)
