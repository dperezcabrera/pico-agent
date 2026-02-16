"""Exception hierarchy for pico-agent.

All pico-agent exceptions inherit from ``AgentError``, making it easy to
catch any framework error with a single ``except AgentError`` clause.
"""


class AgentError(Exception):
    """Base exception for all pico-agent errors."""

    pass


class AgentDisabledError(AgentError):
    """Raised when an agent is invoked but its configuration has ``enabled=False``.

    The error message follows the pattern:

        ``Agent '<name>' is disabled via configuration.``

    Args:
        agent_name: The name of the disabled agent.
    """

    def __init__(self, agent_name: str):
        super().__init__(f"Agent '{agent_name}' is disabled via configuration.")


class AgentConfigurationError(AgentError):
    """Raised for missing or invalid agent / provider configuration.

    Common causes include missing API keys in ``LLMConfig`` or unknown
    provider names.
    """

    pass


class AgentLifecycleError(AgentError):
    """Raised when an operation violates the agent system lifecycle.

    For example, attempting to use the system before it has reached the
    ``READY`` phase.
    """

    pass
