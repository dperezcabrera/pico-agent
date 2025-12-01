import operator
import asyncio
from typing import Any, Protocol, List, Dict, Annotated, TypedDict, Type, TypeVar
from pydantic import BaseModel, Field
from pico_ioc import component, PicoContainer
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from .config import AgentConfig, AgentType
from .registry import AgentConfigService, ToolRegistry
from .interfaces import LLMFactory
from .router import ModelRouter
from .tools import ToolWrapper
from .decorators import TOOL_META_KEY
from .scheduler import PlatformScheduler

T = TypeVar("T")

class VirtualAgent(Protocol):
    def run(self, input: str) -> str: ...
    def run_structured(self, input: str, schema: Type[T]) -> T: ...
    def run_with_args(self, args: Dict[str, Any]) -> str: ...

class TaskItem(BaseModel):
    worker_type: str = Field(description="The type of worker agent to handle this task")
    arguments: Dict[str, Any] = Field(description="The structured arguments/payload for the worker")

class SplitterOutput(BaseModel):
    tasks: List[TaskItem] = Field(description="The list of tasks to be distributed")

class MapReduceState(TypedDict):
    input: str
    tasks: List[TaskItem]
    mapped_results: Annotated[List[str], operator.add]
    final_output: str

class VirtualAgentRunner:
    def __init__(
        self,
        config: AgentConfig,
        tool_registry: ToolRegistry,
        llm_factory: LLMFactory,
        model_router: ModelRouter,
        container: PicoContainer,
        locator: Any,
        scheduler: PlatformScheduler
    ):
        self.config = config
        self.tool_registry = tool_registry
        self.llm_factory = llm_factory
        self.model_router = model_router
        self.container = container
        self.locator = locator
        self.scheduler = scheduler

    def _create_llm(self):
        final_model_name = self.model_router.resolve_model(
            capability=self.config.capability,
            runtime_override=None
        )
        return self.llm_factory.create(
            model_name=final_model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            llm_profile=self.config.llm_profile
        )

    def run(self, input: str) -> str:
        return self.run_with_args({"input": input})

    def run_with_args(self, args: Dict[str, Any]) -> str:
        if not self.config.enabled:
            return "Agent is disabled."

        if self.config.agent_type == AgentType.WORKFLOW:
            input_val = args.get("input", str(args))
            return self._run_workflow(input_val)

        llm = self._create_llm()
        resolved_tools = self._resolve_tools()
        messages = self._build_messages(args)

        if self.config.agent_type == AgentType.REACT:
            return llm.invoke_agent_loop(
                messages,
                resolved_tools,
                self.config.max_iterations
            )
        else:
            return llm.invoke(messages, resolved_tools)

    def run_structured(self, input: str, schema: Type[T]) -> T:
        if not self.config.enabled:
            raise ValueError("Agent is disabled")

        llm = self._create_llm()
        resolved_tools = self._resolve_tools()
        messages = self._build_messages({"input": input})

        return llm.invoke_structured(messages, resolved_tools, schema)

    def _run_workflow(self, input: str) -> str:
        workflow_type = self.config.workflow_config.get("type")
        
        if workflow_type == "map_reduce":
            return self._run_map_reduce(input)
        
        raise ValueError(f"Unknown workflow type: {workflow_type} for agent {self.config.name}")

    def _run_map_reduce(self, input: str) -> str:
        cfg = self.config.workflow_config
        splitter_name = cfg.get("splitter")
        reducer_name = cfg.get("reducer")
        
        mappers_cfg = cfg.get("mappers")
        simple_mapper = cfg.get("mapper")

        if not splitter_name or not reducer_name:
            raise ValueError("Map-Reduce requires 'splitter' and 'reducer'")
        
        if not mappers_cfg and not simple_mapper:
            raise ValueError("Map-Reduce requires either 'mapper' (string) or 'mappers' (dict)")

        workflow = StateGraph(MapReduceState)
        
        async def splitter_node(state: MapReduceState):
            splitter = self.locator.get_agent(splitter_name)
            result: SplitterOutput = splitter.run_structured(state["input"], SplitterOutput)
            return {"tasks": result.tasks}

        async def mapper_node(state: dict):
            task_item: TaskItem = state["task_item"]
            
            if mappers_cfg and isinstance(mappers_cfg, dict):
                worker_name = mappers_cfg.get(task_item.worker_type)
                if not worker_name:
                    worker_name = simple_mapper
            else:
                worker_name = simple_mapper

            if not worker_name:
                 return {"mapped_results": [f"Error: No worker found for type {task_item.worker_type}"]}

            worker = self.locator.get_agent(worker_name)
            
            async with self.scheduler.semaphore:
                result = worker.run_with_args(task_item.arguments)
            
            return {"mapped_results": [result]}

        async def reducer_node(state: MapReduceState):
            reducer = self.locator.get_agent(reducer_name)
            combined_input = "\n\n".join(state["mapped_results"])
            final = reducer.run(combined_input)
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
        
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            return asyncio.run(app.ainvoke({"input": input}))["final_output"]
        else:
            return asyncio.run(app.ainvoke({"input": input}))["final_output"]

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

    def _build_messages(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        messages = []
        if self.config.system_prompt:
            try:
                sys_content = self.config.system_prompt.format(**context)
            except KeyError:
                sys_content = self.config.system_prompt
            messages.append({"role": "system", "content": sys_content})
        
        try:
            user_content = self.config.user_prompt_template.format(**context)
        except KeyError:
            user_content = str(context)
            if "input" in context:
                user_content = context["input"]

        messages.append({"role": "user", "content": user_content})
        return messages

@component
class VirtualAgentManager:
    def __init__(
        self,
        config_service: AgentConfigService,
        tool_registry: ToolRegistry,
        llm_factory: LLMFactory,
        model_router: ModelRouter,
        container: PicoContainer,
        scheduler: PlatformScheduler
    ):
        self.config_service = config_service
        self.tool_registry = tool_registry
        self.llm_factory = llm_factory
        self.model_router = model_router
        self.container = container
        self.scheduler = scheduler

    def create_agent(self, name: str, **kwargs) -> VirtualAgent:
        config = AgentConfig(name=name, **kwargs)
        config_data = config.__dict__.copy()
        if "name" in config_data:
            del config_data["name"]
        self.config_service.update_agent_config(name, **config_data)
        return self.get_agent(name)

    def get_agent(self, name: str) -> VirtualAgent:
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
            scheduler=self.scheduler
        )
