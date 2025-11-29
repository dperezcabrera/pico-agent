from typing import List, Optional, Callable, Type
from .config import AgentConfig, AgentType, AgentCapability

AGENT_META_KEY = "_pico_agent_meta"
IS_AGENT_INTERFACE = "_pico_is_agent_interface"

def agent(
    name: str,
    capability: str = AgentCapability.SMART,
    system_prompt: str = "",
    user_prompt_template: str = "{input}",
    agent_type: AgentType = AgentType.ONE_SHOT,
    max_iterations: int = 5,
    tools: Optional[List[str]] = None,
    agents: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    tracing_enabled: bool = True,
    temperature: float = 0.7,
    llm_profile: Optional[str] = None
) -> Callable[[Type], Type]:

    def decorator(cls_or_proto: Type) -> Type:
        default_config = AgentConfig(
            name=name,
            capability=capability,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            agent_type=agent_type,
            max_iterations=max_iterations,
            tools=tools or [],
            agents=agents or [],
            tags=tags or [],
            tracing_enabled=tracing_enabled,
            temperature=temperature,
            llm_profile=llm_profile
        )

        setattr(cls_or_proto, AGENT_META_KEY, default_config)
        setattr(cls_or_proto, IS_AGENT_INTERFACE, True)
        return cls_or_proto

    return decorator
