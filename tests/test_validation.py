import pytest

from pico_agent.config import AgentCapability, AgentConfig
from pico_agent.validation import AgentValidator, Severity, ValidationIssue, ValidationReport


class TestSeverity:
    def test_warning_value(self):
        assert Severity.WARNING.value == "warning"

    def test_error_value(self):
        assert Severity.ERROR.value == "error"

    def test_is_str_enum(self):
        assert isinstance(Severity.WARNING, str)


class TestValidationIssue:
    def test_create_issue(self):
        issue = ValidationIssue(field="name", message="Name is required", severity=Severity.ERROR)
        assert issue.field == "name"
        assert issue.message == "Name is required"
        assert issue.severity == Severity.ERROR


class TestValidationReport:
    def test_valid_report(self):
        report = ValidationReport(valid=True, issues=[])
        assert report.valid is True
        assert report.issues == []
        assert report.has_errors is False

    def test_report_with_warnings_only(self):
        issues = [ValidationIssue("field1", "warning message", Severity.WARNING)]
        report = ValidationReport(valid=True, issues=issues)
        assert report.valid is True
        assert report.has_errors is False

    def test_report_with_errors(self):
        issues = [ValidationIssue("field1", "error message", Severity.ERROR)]
        report = ValidationReport(valid=False, issues=issues)
        assert report.valid is False
        assert report.has_errors is True

    def test_has_errors_with_mixed_issues(self):
        issues = [
            ValidationIssue("field1", "warning", Severity.WARNING),
            ValidationIssue("field2", "error", Severity.ERROR),
        ]
        report = ValidationReport(valid=False, issues=issues)
        assert report.has_errors is True


class TestAgentValidator:
    @pytest.fixture
    def validator(self):
        return AgentValidator()

    def test_validate_valid_config(self, validator, sample_agent_config):
        report = validator.validate(sample_agent_config)
        assert report.valid is True
        assert not report.has_errors

    def test_validate_empty_name(self, validator):
        config = AgentConfig(name="")
        report = validator.validate(config)
        assert report.valid is False
        assert any(i.field == "name" and i.severity == Severity.ERROR for i in report.issues)

    def test_validate_whitespace_name(self, validator):
        config = AgentConfig(name="   ")
        report = validator.validate(config)
        assert report.valid is False
        assert any(i.field == "name" for i in report.issues)

    def test_validate_negative_temperature(self, validator):
        config = AgentConfig(name="test", temperature=-0.5)
        report = validator.validate(config)
        assert report.valid is False
        assert any(i.field == "temperature" and i.severity == Severity.ERROR for i in report.issues)

    def test_validate_temperature_too_high(self, validator):
        config = AgentConfig(name="test", temperature=2.5)
        report = validator.validate(config)
        assert report.valid is False
        assert any(i.field == "temperature" for i in report.issues)

    def test_validate_high_temperature_warning(self, validator):
        config = AgentConfig(name="test", temperature=1.5, system_prompt="test")
        report = validator.validate(config)
        assert report.valid is True
        assert any(i.field == "temperature" and i.severity == Severity.WARNING for i in report.issues)

    def test_validate_empty_system_prompt_warning(self, validator):
        config = AgentConfig(name="test", system_prompt="")
        report = validator.validate(config)
        assert any(i.field == "system_prompt" and i.severity == Severity.WARNING for i in report.issues)

    def test_validate_missing_capability(self, validator):
        config = AgentConfig(name="test", capability="")
        report = validator.validate(config)
        assert report.valid is False

    def test_validate_boundary_temperature_values(self, validator):
        # Temperature at lower boundary (0.0)
        config = AgentConfig(name="test", temperature=0.0, system_prompt="test")
        report = validator.validate(config)
        assert report.valid is True

        # Temperature at upper boundary (2.0)
        config = AgentConfig(name="test", temperature=2.0, system_prompt="test")
        report = validator.validate(config)
        assert report.valid is True
        # Should have warning for temperature > 1.0
        assert any(i.field == "temperature" and i.severity == Severity.WARNING for i in report.issues)


class TestValidatorIntegration:
    def test_multiple_issues_detected(self):
        validator = AgentValidator()
        config = AgentConfig(name="", temperature=-1.0, capability="", system_prompt="")
        report = validator.validate(config)
        assert report.valid is False
        assert len(report.issues) >= 2  # At least name and temperature errors
