"""Configuration dataclasses and enumerations for pico-agent.

Defines the core configuration types used throughout the framework:
``AgentType``, ``AgentCapability``, ``AgentConfig``, ``ToolConfig``,
and ``LLMConfig``.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentType(str, Enum):
    """Execution strategy for an agent.

    Determines how the agent processes requests and interacts with tools.

    Attributes:
        ONE_SHOT: Single LLM call with no tool loop (default).
        REACT: Iterative ReAct tool loop via LangGraph, up to
            ``max_iterations`` rounds.
        WORKFLOW: Custom workflow execution (e.g., map-reduce).
    """

    ONE_SHOT = "one_shot"
    REACT = "react"
    WORKFLOW = "workflow"


class AgentCapability:
    """Abstract capability labels mapped to concrete models by ``ModelRouter``.

    Use these constants in the ``@agent`` decorator to declare what kind of
    model an agent needs.  The ``ModelRouter`` translates these labels into
    provider-specific model names at runtime, allowing you to swap models
    globally without touching agent definitions.

    Attributes:
        FAST: Optimised for low latency (default model: ``gpt-5-mini``).
        SMART: Balanced quality and cost (default model: ``gpt-5.1``).
        REASONING: Advanced reasoning tasks (default model: ``gemini-3-pro``).
        VISION: Vision / multimodal support (default model: ``gpt-4o``).
        CODING: Code generation (default model: ``claude-3-5-sonnet``).

    Example:
        >>> from pico_agent import agent, AgentCapability
        >>> @agent(name="fast_bot", capability=AgentCapability.FAST)
        ... class FastBot(Protocol):
        ...     def run(self, q: str) -> str: ...
    """

    FAST = "fast"
    SMART = "smart"
    REASONING = "reasoning"
    VISION = "vision"
    CODING = "coding"


@dataclass
class AgentConfig:
    """Complete configuration for a single agent.

    Instances are created automatically by the ``@agent`` decorator and stored
    in ``LocalAgentRegistry``.  The ``AgentConfigService`` merges local, remote
    (central), and runtime overrides to produce the final effective config.

    Args:
        name: Unique agent identifier (required).
        system_prompt: System-level prompt sent to the LLM.
        user_prompt_template: Template for the user message.  Use ``{input}``
            or any key matching the method signature.
        description: Human-readable description; used as ``AgentAsTool``
            description when the agent is exposed as a tool.
        capability: ``AgentCapability`` constant that the ``ModelRouter``
            resolves to a concrete model name.
        enabled: Whether the agent is active.  Disabled agents raise
            ``AgentDisabledError``.
        agent_type: Execution strategy (``ONE_SHOT``, ``REACT``, or
            ``WORKFLOW``).
        max_iterations: Maximum ReAct loop iterations (only relevant for
            ``REACT`` agents).
        tools: List of tool names to attach to this agent.
        agents: List of child agent names that will be wrapped as
            ``AgentAsTool`` instances.
        tags: Tags used for dynamic tool lookup via ``ToolRegistry``.
        tracing_enabled: Whether ``TraceService`` records runs for this agent.
        temperature: LLM sampling temperature (0.0 -- 2.0).
        max_tokens: Maximum tokens in the LLM response, or ``None`` for the
            provider default.
        llm_profile: Named API-key / base-URL profile in ``LLMConfig``.
        workflow_config: Extra parameters for ``WORKFLOW`` agents (e.g.,
            ``{"type": "map_reduce", "splitter": "...", "reducer": "..."}``).
    """

    name: str
    system_prompt: str = ""
    user_prompt_template: str = "{input}"
    description: str = ""
    capability: str = AgentCapability.SMART
    enabled: bool = True
    agent_type: AgentType = AgentType.ONE_SHOT
    max_iterations: int = 5
    tools: List[str] = field(default_factory=list)
    agents: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    tracing_enabled: bool = True
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    llm_profile: Optional[str] = None
    workflow_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolConfig:
    """Metadata for a tool, created by the ``@tool`` decorator.

    Args:
        name: Unique tool identifier shown to the LLM for tool selection.
        description: Human-readable description the LLM uses to decide when
            to invoke this tool.
    """

    name: str
    description: str


@dataclass
class LLMConfig:
    """Centralised API-key and base-URL store for all LLM providers.

    ``AgentLocator`` registers a default (empty) ``LLMConfig`` singleton via
    ``@provides``.  To populate it with your credentials, use ``@configure``
    on a component method that receives ``LLMConfig`` as a parameter.  Do
    **not** register your own ``LLMConfig`` with ``@factory`` + ``@provides``
    -- that would conflict with the singleton already provided by
    ``AgentInfrastructureFactory``.

    Args:
        api_keys: Mapping of provider name (or profile name) to API key.
            Standard keys: ``"openai"``, ``"anthropic"``, ``"google"``,
            ``"azure"``, ``"deepseek"``, ``"qwen"``.
        base_urls: Mapping of provider name (or profile name) to base URL
            override.

    Example:
        >>> from pico_ioc import component, configure
        >>> from pico_agent import LLMConfig
        >>> @component
        ... class AppConfig:
        ...     @configure
        ...     def setup(self, llm: LLMConfig):
        ...         llm.api_keys["openai"] = "sk-..."
    """

    api_keys: Dict[str, str] = field(default_factory=dict)
    base_urls: Dict[str, str] = field(default_factory=dict)
