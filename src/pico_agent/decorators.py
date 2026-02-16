"""Decorators for declaring agents and tools.

Provides the ``@agent`` and ``@tool`` class decorators that attach metadata
used by ``AgentScanner`` and ``ToolScanner`` for automatic discovery.
"""

from typing import Callable, List, Optional, Type

from .config import AgentCapability, AgentConfig, AgentType, ToolConfig

AGENT_META_KEY = "_pico_agent_meta"
"""str: Attribute name where ``AgentConfig`` metadata is stored on a decorated class."""

TOOL_META_KEY = "_pico_tool_meta"
"""str: Attribute name where ``ToolConfig`` metadata is stored on a decorated class."""

IS_AGENT_INTERFACE = "_pico_is_agent_interface"
"""str: Boolean flag attribute set on classes decorated with ``@agent``."""


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
    llm_profile: Optional[str] = None,
) -> Callable[[Type], Type]:
    """Declare a Protocol class as a pico-agent.

    Attaches an ``AgentConfig`` instance (``AGENT_META_KEY``) and the
    ``IS_AGENT_INTERFACE`` flag to the decorated class so that
    ``AgentScanner`` can discover and register it automatically.

    Args:
        name: Unique agent identifier (required).
        capability: ``AgentCapability`` constant used by ``ModelRouter`` to
            select a concrete model.
        system_prompt: System-level prompt sent to the LLM.
        description: Human-readable description.  If empty, falls back to the
            first line of the class docstring.
        user_prompt_template: Template for the user message.  Placeholders
            (e.g., ``{input}``) are filled from method arguments.
        agent_type: Execution strategy -- ``ONE_SHOT``, ``REACT``, or
            ``WORKFLOW``.
        max_iterations: Maximum ReAct loop iterations (``REACT`` only).
        tools: Tool names to attach to this agent.
        agents: Child agent names wrapped as ``AgentAsTool``.
        tags: Tags for dynamic tool lookup via ``ToolRegistry``.
        tracing_enabled: Whether ``TraceService`` records runs.
        temperature: LLM sampling temperature (0.0 -- 2.0).
        llm_profile: Named profile in ``LLMConfig`` for API key / base URL.

    Returns:
        A class decorator that sets agent metadata on the target class.

    Example:
        >>> from typing import Protocol
        >>> from pico_agent import agent, AgentCapability, AgentType
        >>> @agent(
        ...     name="summarizer",
        ...     capability=AgentCapability.SMART,
        ...     system_prompt="Summarize the following text.",
        ...     agent_type=AgentType.ONE_SHOT,
        ... )
        ... class Summarizer(Protocol):
        ...     def summarize(self, text: str) -> str: ...
    """

    def decorator(cls_or_proto: Type) -> Type:
        final_desc = description
        if not final_desc and cls_or_proto.__doc__:
            final_desc = cls_or_proto.__doc__.strip().split("\n")[0]

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
            llm_profile=llm_profile,
        )

        setattr(cls_or_proto, AGENT_META_KEY, default_config)
        setattr(cls_or_proto, IS_AGENT_INTERFACE, True)
        return cls_or_proto

    return decorator


def tool(name: str, description: str) -> Callable[[Type], Type]:
    """Declare a class as a pico-agent tool.

    Attaches a ``ToolConfig`` instance (``TOOL_META_KEY``) to the decorated
    class so that ``ToolScanner`` can discover and register it.  The class
    must implement one of ``__call__``, ``run``, ``execute``, or ``invoke``.

    Args:
        name: Unique tool identifier shown to the LLM.
        description: Human-readable description the LLM uses to decide when
            to invoke this tool.

    Returns:
        A class decorator that sets tool metadata on the target class.

    Example:
        >>> from pico_ioc import component
        >>> from pico_agent import tool
        >>> @tool(name="calculator", description="Evaluate math expressions")
        ... @component
        ... class Calculator:
        ...     def run(self, expression: str) -> str:
        ...         return str(eval(expression))
    """

    def decorator(cls: Type) -> Type:
        config = ToolConfig(name=name, description=description)
        setattr(cls, TOOL_META_KEY, config)
        return cls

    return decorator
