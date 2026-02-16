"""Protocol interfaces that define the contracts for agents, LLMs, and configuration clients.

All core abstractions are expressed as ``typing.Protocol`` classes so that
any conforming implementation can be used without explicit inheritance.
"""

from typing import Any, Awaitable, Dict, List, Optional, Protocol, Type, TypeVar

T = TypeVar("T")


class Agent(Protocol):
    """Standard interface for an agent produced by pico-agent.

    Agent Protocol classes decorated with ``@agent`` do not need to
    inherit from this Protocol; it exists as a reference contract.
    """

    def run(self, input: str) -> str:
        """Execute the agent synchronously.

        Args:
            input: The user message / task description.

        Returns:
            The agent's text response.
        """
        ...

    async def arun(self, input: str) -> str:
        """Execute the agent asynchronously.

        Args:
            input: The user message / task description.

        Returns:
            The agent's text response.
        """
        ...

    def run_structured(self, input: str, schema: Type[T]) -> T:
        """Execute the agent and parse the response into a Pydantic model.

        Args:
            input: The user message / task description.
            schema: A ``pydantic.BaseModel`` subclass that defines the
                expected response structure.

        Returns:
            An instance of *schema* populated from the LLM response.
        """
        ...

    async def arun_structured(self, input: str, schema: Type[T]) -> T:
        """Async variant of ``run_structured``.

        Args:
            input: The user message / task description.
            schema: A ``pydantic.BaseModel`` subclass for structured output.

        Returns:
            An instance of *schema* populated from the LLM response.
        """
        ...


class CentralConfigClient(Protocol):
    """Protocol for retrieving and persisting agent configuration remotely.

    The default implementation (``NoOpCentralClient``) returns ``None`` for
    all lookups, meaning only local and runtime config is used.  Provide a
    custom implementation (e.g., backed by a database or API) to enable
    central configuration management.
    """

    def get_agent_config(self, name: str) -> Optional[Any]:
        """Fetch the remote configuration for an agent.

        Args:
            name: The agent's unique identifier.

        Returns:
            An ``AgentConfig`` if one exists remotely, otherwise ``None``.
        """
        ...

    def upsert_agent_config(self, config: Any) -> None:
        """Create or update the remote configuration for an agent.

        Args:
            config: The ``AgentConfig`` to persist.
        """
        ...


class LLM(Protocol):
    """Protocol for a language-model adapter used by agent proxies.

    ``LangChainAdapter`` is the built-in implementation.
    """

    def invoke(self, messages: List[Dict[str, str]], tools: List[Any]) -> str:
        """Send messages to the LLM and return the text response.

        Args:
            messages: List of message dicts with ``"role"`` and ``"content"``
                keys.
            tools: LangChain-compatible tool instances bound to the model.

        Returns:
            The LLM's text response.
        """
        ...

    def invoke_structured(self, messages: List[Dict[str, str]], tools: List[Any], output_schema: Type[Any]) -> Any:
        """Send messages and parse the response into a structured schema.

        Args:
            messages: List of message dicts.
            tools: LangChain-compatible tool instances.
            output_schema: A ``pydantic.BaseModel`` subclass for structured
                output.

        Returns:
            An instance of *output_schema*.
        """
        ...

    def invoke_agent_loop(
        self,
        messages: List[Dict[str, str]],
        tools: List[Any],
        max_iterations: int,
        output_schema: Optional[Type[Any]] = None,
    ) -> Any:
        """Run a ReAct-style tool loop via LangGraph.

        Args:
            messages: List of message dicts.
            tools: LangChain-compatible tool instances.
            max_iterations: Maximum number of reasoning iterations.
            output_schema: Optional Pydantic model for structured final
                output.

        Returns:
            The final text response, or an instance of *output_schema* if
            provided.
        """
        ...


class LLMFactory(Protocol):
    """Protocol for creating ``LLM`` instances from model parameters.

    ``LangChainLLMFactory`` is the built-in implementation that supports
    OpenAI, Azure, Anthropic, Google, DeepSeek, and Qwen.
    """

    def create(
        self, model_name: str, temperature: float, max_tokens: Optional[int], llm_profile: Optional[str] = None
    ) -> LLM:
        """Create an ``LLM`` instance for the given model.

        Args:
            model_name: Model identifier, optionally prefixed with a provider
                (e.g., ``"openai:gpt-5-mini"``).
            temperature: Sampling temperature (0.0 -- 2.0).
            max_tokens: Maximum response tokens, or ``None`` for the provider
                default.
            llm_profile: Named profile in ``LLMConfig`` for API key / base
                URL selection.

        Returns:
            A configured ``LLM`` instance ready for invocation.
        """
        ...
