"""Virtual (config-only) agents and the map-reduce workflow engine.

``VirtualAgentRunner`` executes agent configurations that have no
corresponding Protocol class (e.g., agents defined via YAML or created at
runtime with ``VirtualAgentManager``).  ``VirtualAgentManager`` is a
component that creates and retrieves virtual agents programmatically.
"""

import asyncio
import operator
from typing import Annotated, Any, Dict, List, Protocol, Type, TypedDict, TypeVar

from langgraph.graph import END, StateGraph
from langgraph.types import Send
from pico_ioc import PicoContainer, component
from pydantic import BaseModel, Field

from .config import AgentConfig, AgentType
from .decorators import TOOL_META_KEY
from .interfaces import LLMFactory
from .messages import build_messages
from .registry import AgentConfigService, ToolRegistry
from .router import ModelRouter
from .scheduler import PlatformScheduler
from .tools import ToolWrapper

T = TypeVar("T")


class VirtualAgent(Protocol):
    """Protocol for virtual agents (config-only, no Protocol class).

    Both ``VirtualAgentRunner`` and dynamically created agents conform to
    this interface.
    """

    def run(self, input: str) -> str:
        """Execute the agent synchronously.

        Args:
            input: The user message.

        Returns:
            The agent's text response.
        """
        ...

    async def arun(self, input: str) -> str:
        """Execute the agent asynchronously.

        Args:
            input: The user message.

        Returns:
            The agent's text response.
        """
        ...

    def run_structured(self, input: str, schema: Type[T]) -> T:
        """Execute the agent and parse the response into a Pydantic model.

        Args:
            input: The user message.
            schema: A ``pydantic.BaseModel`` subclass.

        Returns:
            An instance of *schema*.
        """
        ...

    def run_with_args(self, args: Dict[str, Any]) -> str:
        """Execute the agent with a dictionary of arguments.

        Args:
            args: Key-value pairs used to fill prompt templates.

        Returns:
            The agent's text response.
        """
        ...


class TaskItem(BaseModel):
    """A single work item produced by the splitter in a map-reduce workflow.

    Attributes:
        worker_type: Key used to select a mapper agent from ``mappers`` config.
        arguments: Structured arguments forwarded to the mapper agent.
    """

    worker_type: str = Field(description="The type of worker agent to handle this task")
    arguments: Dict[str, Any] = Field(description="The structured arguments/payload for the worker")


class SplitterOutput(BaseModel):
    """Structured output expected from the splitter agent in a map-reduce workflow.

    Attributes:
        tasks: The list of ``TaskItem`` objects to distribute to mappers.
    """

    tasks: List[TaskItem] = Field(description="The list of tasks to be distributed")


class MapReduceState(TypedDict):
    """LangGraph state for the map-reduce workflow.

    Attributes:
        input: The original user input.
        tasks: Tasks produced by the splitter.
        mapped_results: Accumulated results from mapper agents (additive).
        final_output: The reducer agent's final combined output.
    """

    input: str
    tasks: List[TaskItem]
    mapped_results: Annotated[List[str], operator.add]
    final_output: str


class VirtualAgentRunner:
    """Executes agent configurations without a Protocol class.

    Supports ``ONE_SHOT``, ``REACT``, and ``WORKFLOW`` (map-reduce) agent
    types.  Created by ``AgentLocator`` for agents that exist only as
    configuration (e.g., YAML-defined or runtime-created agents).

    Args:
        config: The agent's ``AgentConfig``.
        tool_registry: Registry for tool lookup.
        llm_factory: Factory for creating LLM instances.
        model_router: Capability-to-model router.
        container: The pico-ioc container.
        locator: ``AgentLocator`` for resolving child agents.
        scheduler: Concurrency scheduler for async map-reduce.
    """

    def __init__(
        self,
        config: AgentConfig,
        tool_registry: ToolRegistry,
        llm_factory: LLMFactory,
        model_router: ModelRouter,
        container: PicoContainer,
        locator: Any,
        scheduler: PlatformScheduler,
    ):
        self.config = config
        self.tool_registry = tool_registry
        self.llm_factory = llm_factory
        self.model_router = model_router
        self.container = container
        self.locator = locator
        self.scheduler = scheduler

    def _create_llm(self):
        final_model_name = self.model_router.resolve_model(capability=self.config.capability, runtime_override=None)
        return self.llm_factory.create(
            model_name=final_model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            llm_profile=self.config.llm_profile,
        )

    def run(self, input: str) -> str:
        """Execute the agent synchronously.

        Args:
            input: The user message.

        Returns:
            The agent's text response.
        """
        return self.run_with_args({"input": input})

    async def arun(self, input: str) -> str:
        """Execute the agent asynchronously.

        For ``WORKFLOW`` agents, runs the async workflow directly.  For
        other types, delegates to ``asyncio.to_thread``.

        Args:
            input: The user message.

        Returns:
            The agent's text response.
        """
        if self.config.agent_type == AgentType.WORKFLOW:
            return await self._arun_workflow({"input": input})

        return await asyncio.to_thread(self.run, input)

    def run_with_args(self, args: Dict[str, Any]) -> str:
        """Execute the agent with a dictionary of arguments.

        Args:
            args: Key-value pairs used to fill prompt templates.

        Returns:
            The agent's text response, or ``"Agent is disabled."`` if the
            agent is not enabled.

        Raises:
            RuntimeError: If a ``WORKFLOW`` agent is called synchronously
                from within an already-running async event loop.  Message:
                ``"Cannot call sync run() from inside an async loop. Use await agent.arun() instead."``
        """
        if not self.config.enabled:
            return "Agent is disabled."

        if self.config.agent_type == AgentType.WORKFLOW:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                raise RuntimeError("Cannot call sync run() from inside an async loop. Use await agent.arun() instead.")

            return asyncio.run(self._arun_workflow(args))

        llm = self._create_llm()
        resolved_tools = self._resolve_tools()
        messages = build_messages(self.config, args)

        if self.config.agent_type == AgentType.REACT:
            return llm.invoke_agent_loop(messages, resolved_tools, self.config.max_iterations)
        else:
            return llm.invoke(messages, resolved_tools)

    def run_structured(self, input: str, schema: Type[T]) -> T:
        """Execute the agent and parse the response into a Pydantic model.

        Args:
            input: The user message.
            schema: A ``pydantic.BaseModel`` subclass.

        Returns:
            An instance of *schema* populated from the LLM response.

        Raises:
            ValueError: If the agent is disabled.
        """
        if not self.config.enabled:
            raise ValueError("Agent is disabled")

        llm = self._create_llm()
        resolved_tools = self._resolve_tools()
        messages = build_messages(self.config, {"input": input})

        return llm.invoke_structured(messages, resolved_tools, schema)

    async def _arun_workflow(self, args: Dict[str, Any]) -> str:
        workflow_type = self.config.workflow_config.get("type")
        input_val = args.get("input", str(args))

        if workflow_type == "map_reduce":
            return await self._arun_map_reduce(input_val)

        raise ValueError(f"Unknown workflow type: {workflow_type}")

    async def _arun_map_reduce(self, input: str) -> str:
        cfg = self.config.workflow_config
        splitter_name = cfg.get("splitter")
        reducer_name = cfg.get("reducer")
        mappers_cfg = cfg.get("mappers")
        simple_mapper = cfg.get("mapper")

        if not splitter_name or not reducer_name:
            raise ValueError("Map-Reduce requires 'splitter' and 'reducer'")

        workflow = StateGraph(MapReduceState)

        async def splitter_node(state: MapReduceState):
            splitter = self.locator.get_agent(splitter_name)
            result: SplitterOutput = splitter.run_structured(state["input"], SplitterOutput)
            return {"tasks": result.tasks}

        async def mapper_node(state: dict):
            task_item: TaskItem = state["task_item"]

            if mappers_cfg and isinstance(mappers_cfg, dict):
                worker_name = mappers_cfg.get(task_item.worker_type) or simple_mapper
            else:
                worker_name = simple_mapper

            if not worker_name:
                return {"mapped_results": ["Error: No worker found"]}

            worker = self.locator.get_agent(worker_name)

            async with self.scheduler.semaphore:
                result = await asyncio.to_thread(worker.run_with_args, task_item.arguments)

            return {"mapped_results": [result]}

        async def reducer_node(state: MapReduceState):
            reducer = self.locator.get_agent(reducer_name)
            combined_input = "\n\n".join(state["mapped_results"])
            final = await asyncio.to_thread(reducer.run, combined_input)
            return {"final_output": final}

        def distribute_tasks(state: MapReduceState):
            return [Send("mapper", {"task_item": task}) for task in state["tasks"]]

        workflow.add_node("splitter", splitter_node)
        workflow.add_node("mapper", mapper_node)
        workflow.add_node("reducer", reducer_node)

        workflow.set_entry_point("splitter")
        workflow.add_conditional_edges("splitter", distribute_tasks)
        workflow.add_edge("mapper", "reducer")
        workflow.add_edge("reducer", END)

        app = workflow.compile()
        result = await app.ainvoke({"input": input})
        return result["final_output"]

    def _resolve_tools(self) -> List[Any]:
        final_tools = []
        for tool_name in self.config.tools:
            tool_instance = None
            if self.container.has(tool_name):
                tool_instance = self.container.get(tool_name)
            elif self.tool_registry.get_tool(tool_name):
                tool_ref = self.tool_registry.get_tool(tool_name)
                tool_instance = tool_ref() if isinstance(tool_ref, type) else tool_ref

            if tool_instance:
                if hasattr(tool_instance, "args_schema") and hasattr(tool_instance, "name"):
                    final_tools.append(tool_instance)
                elif hasattr(type(tool_instance), TOOL_META_KEY):
                    tool_config = getattr(type(tool_instance), TOOL_META_KEY)
                    final_tools.append(ToolWrapper(tool_instance, tool_config))
                else:
                    final_tools.append(tool_instance)
        return final_tools


@component
class VirtualAgentManager:
    """Creates and manages virtual agents programmatically at runtime.

    Virtual agents are config-only agents that do not require a Protocol
    class.  Use ``create_agent()`` to define a new agent with inline
    parameters, or ``get_agent()`` to retrieve an existing one.

    Args:
        config_service: Service for storing runtime agent configurations.
        tool_registry: Registry for tool lookup.
        llm_factory: Factory for creating LLM instances.
        model_router: Capability-to-model router.
        container: The pico-ioc container.
        scheduler: Concurrency scheduler for async operations.

    Example:
        >>> manager = container.get(VirtualAgentManager)
        >>> agent = manager.create_agent(
        ...     "greeter",
        ...     system_prompt="You greet users warmly.",
        ...     capability="fast",
        ... )
        >>> result = agent.run("Hello!")
    """

    def __init__(
        self,
        config_service: AgentConfigService,
        tool_registry: ToolRegistry,
        llm_factory: LLMFactory,
        model_router: ModelRouter,
        container: PicoContainer,
        scheduler: PlatformScheduler,
    ):
        self.config_service = config_service
        self.tool_registry = tool_registry
        self.llm_factory = llm_factory
        self.model_router = model_router
        self.container = container
        self.scheduler = scheduler

    def create_agent(self, name: str, **kwargs) -> VirtualAgent:
        """Create a new virtual agent and register its configuration.

        Args:
            name: Unique agent identifier.
            **kwargs: Any ``AgentConfig`` fields (e.g., ``system_prompt``,
                ``capability``, ``tools``, ``agent_type``).

        Returns:
            A ``VirtualAgentRunner`` conforming to the ``VirtualAgent``
            protocol.
        """
        config = AgentConfig(name=name, **kwargs)
        config_data = config.__dict__.copy()
        if "name" in config_data:
            del config_data["name"]
        self.config_service.update_agent_config(name, **config_data)
        return self.get_agent(name)

    def get_agent(self, name: str) -> VirtualAgent:
        """Retrieve (or re-create) a virtual agent by name.

        Args:
            name: The agent identifier previously used with ``create_agent()``
                or registered via ``AgentConfigService``.

        Returns:
            A ``VirtualAgentRunner`` for the named agent.

        Raises:
            ValueError: If no configuration exists for the given name.
        """
        from .locator import AgentLocator

        locator = self.container.get(AgentLocator)
        config = self.config_service.get_config(name)
        return VirtualAgentRunner(
            config=config,
            tool_registry=self.tool_registry,
            llm_factory=self.llm_factory,
            model_router=self.model_router,
            container=self.container,
            locator=locator,
            scheduler=self.scheduler,
        )
