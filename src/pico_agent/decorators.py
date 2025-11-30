from typing import List, Optional, Callable, Type
from .config import AgentConfig, AgentType, AgentCapability, ToolConfig

AGENT_META_KEY = "_pico_agent_meta"
TOOL_META_KEY = "_pico_tool_meta"
IS_AGENT_INTERFACE = "_pico_is_agent_interface"

def agent(
    name: str,
    capability: str = AgentCapability.SMART,
    system_prompt: str = "",
    description: str = "",
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
        final_desc = description
        if not final_desc and cls_or_proto.__doc__:
            final_desc = cls_or_proto.__doc__.strip().split('\n')[0]

        default_config = AgentConfig(
            name=name,
            capability=capability,
            system_prompt=system_prompt,
            description=final_desc,
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

def tool(name: str, description: str) -> Callable[[Type], Type]:
    def decorator(cls: Type) -> Type:
        config = ToolConfig(name=name, description=description)
        setattr(cls, TOOL_META_KEY, config)
        return cls
    return decorator
