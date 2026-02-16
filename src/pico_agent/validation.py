"""Agent configuration validation.

``AgentValidator`` checks an ``AgentConfig`` for missing or invalid fields
and returns a ``ValidationReport`` with categorised issues.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List

from .config import AgentConfig


class Severity(str, Enum):
    """Severity level for a validation issue.

    Attributes:
        WARNING: Non-fatal issue (e.g., empty system prompt).
        ERROR: Fatal issue that prevents the agent from running.
    """

    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationIssue:
    """A single validation finding.

    Args:
        field: The ``AgentConfig`` field name that triggered the issue.
        message: Human-readable description of the problem.
        severity: ``WARNING`` or ``ERROR``.
    """

    field: str
    message: str
    severity: Severity


@dataclass
class ValidationReport:
    """Result of validating an ``AgentConfig``.

    Args:
        valid: ``True`` if no ``ERROR``-level issues were found.
        issues: List of ``ValidationIssue`` instances.
    """

    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Return ``True`` if any issue has ``Severity.ERROR``."""
        return any(i.severity == Severity.ERROR for i in self.issues)


class AgentValidator:
    """Validates ``AgentConfig`` instances for correctness.

    Checks performed:

    - ``name`` must not be empty.
    - ``capability`` must be defined.
    - ``temperature`` must be between 0.0 and 2.0 (warning if > 1.0).
    - ``system_prompt`` should not be empty (warning).
    """

    def validate(self, config: AgentConfig) -> ValidationReport:
        """Validate an agent configuration.

        Args:
            config: The ``AgentConfig`` to validate.

        Returns:
            A ``ValidationReport`` with ``valid=True`` if no errors were
            found, and a list of ``ValidationIssue`` items.
        """
        issues = []

        if not config.name or len(config.name.strip()) == 0:
            issues.append(ValidationIssue("name", "Agent name cannot be empty", Severity.ERROR))

        if not config.capability:
            issues.append(ValidationIssue("capability", "Agent capability must be defined", Severity.ERROR))

        if not (0.0 <= config.temperature <= 2.0):
            issues.append(ValidationIssue("temperature", "Temperature must be between 0.0 and 2.0", Severity.ERROR))
        elif config.temperature > 1.0:
            issues.append(
                ValidationIssue("temperature", "High temperature (>1.0) may cause hallucinations", Severity.WARNING)
            )

        if not config.system_prompt:
            issues.append(ValidationIssue("system_prompt", "System prompt is empty", Severity.WARNING))

        return ValidationReport(valid=not any(i.severity == Severity.ERROR for i in issues), issues=issues)
