from dataclasses import dataclass, field
from typing import List
from enum import Enum
from .config import AgentConfig

class Severity(str, Enum):
    WARNING = "warning"
    ERROR = "error"

@dataclass
class ValidationIssue:
    field: str
    message: str
    severity: Severity

@dataclass
class ValidationReport:
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == Severity.ERROR for i in self.issues)

class AgentValidator:
    def validate(self, config: AgentConfig) -> ValidationReport:
        issues = []

        if not config.name or len(config.name.strip()) == 0:
            issues.append(ValidationIssue("name", "Agent name cannot be empty", Severity.ERROR))
        
        if not config.capability:
            issues.append(ValidationIssue("capability", "Agent capability must be defined", Severity.ERROR))
        
        if not (0.0 <= config.temperature <= 2.0):
            issues.append(ValidationIssue("temperature", "Temperature must be between 0.0 and 2.0", Severity.ERROR))
        elif config.temperature > 1.0:
            issues.append(ValidationIssue("temperature", "High temperature (>1.0) may cause hallucinations", Severity.WARNING))

        if not config.system_prompt:
            issues.append(ValidationIssue("system_prompt", "System prompt is empty", Severity.WARNING))

        return ValidationReport(valid=not any(i.severity == Severity.ERROR for i in issues), issues=issues)
