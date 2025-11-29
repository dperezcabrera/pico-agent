class AgentError(Exception):
    pass

class AgentDisabledError(AgentError):
    def __init__(self, agent_name: str):
        super().__init__(f"Agent '{agent_name}' is disabled via configuration.")

class AgentConfigurationError(AgentError):
    pass
