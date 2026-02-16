import asyncio
import inspect
from typing import Any, Dict, List, Optional, Type, get_type_hints

from pico_ioc import PicoContainer, component
from pydantic import BaseModel

from .config import AgentConfig, AgentType
from .decorators import TOOL_META_KEY
from .exceptions import AgentDisabledError
from .interfaces import LLMFactory
from .logging import get_logger
from .messages import build_messages
from .registry import AgentConfigService, ToolRegistry
from .router import ModelRouter
from .tools import AgentAsTool, ToolWrapper
from .tracing import TraceService

logger = get_logger(__name__)


@component
class TracedAgentProxy:
    def __init__(
        self,
        config_service: AgentConfigService,
        tool_registry: ToolRegistry,
        llm_factory: LLMFactory,
        model_router: ModelRouter,
    ):
        self.config_service = config_service
        self.tool_registry = tool_registry
        self.llm_factory = llm_factory
        self.model_router = model_router

    def execute_agent(self, agent_name: str, user_input: str) -> Any:
        config = self.config_service.get_config(agent_name)

        if not config.enabled:
            raise AgentDisabledError(agent_name)

        final_model_name = self.model_router.resolve_model(capability=config.capability, runtime_override=None)

        llm = self.llm_factory.create(
            model_name=final_model_name, temperature=config.temperature, max_tokens=config.max_tokens
        )

        resolved_tools = []
        for tool_name in config.tools:
            t = self.tool_registry.get_tool(tool_name)
            if t:
                resolved_tools.append(t)

        dynamic = self.tool_registry.get_dynamic_tools(config.tags)
        for dt in dynamic:
            if dt not in resolved_tools:
                resolved_tools.append(dt)

        messages = []
        if config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})

        messages.append({"role": "user", "content": user_input})

        if config.agent_type == AgentType.REACT:
            return llm.invoke_agent_loop(messages, resolved_tools, config.max_iterations)
        else:
            return llm.invoke(messages, resolved_tools)


class DynamicAgentProxy:
    def __init__(
        self,
        agent_name: str,
        protocol_cls: Optional[Type],
        config_service: AgentConfigService,
        tool_registry: ToolRegistry,
        llm_factory: LLMFactory,
        model_router: ModelRouter,
        container: PicoContainer,
        locator: Any = None,
    ):
        self.agent_name = agent_name
        self.protocol_cls = protocol_cls
        self.config_service = config_service
        self.tool_registry = tool_registry
        self.llm_factory = llm_factory
        self.model_router = model_router
        self.container = container
        self.locator = locator

        self.tracer = None
        if self.container.has(TraceService):
            self.tracer = self.container.get(TraceService)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        if not self.protocol_cls:
            raise AttributeError(f"Virtual Agent '{self.agent_name}' has no protocol definition.")

        if not hasattr(self.protocol_cls, name):
            raise AttributeError(f"Agent {self.agent_name} has no method {name}")

        method_ref = getattr(self.protocol_cls, name)

        if not callable(method_ref):
            return method_ref

        method_sig = inspect.signature(method_ref)

        params = list(method_sig.parameters.values())
        if params and params[0].name == "self":
            method_sig = method_sig.replace(parameters=params[1:])

        type_hints = get_type_hints(method_ref)
        return_type = type_hints.get("return", str)

        return self._create_method_wrapper(method_ref, method_sig, return_type)

    def _create_method_wrapper(self, method_ref, method_sig, return_type):
        def method_wrapper(*args, **kwargs):
            input_context = self._extract_input_context(method_sig, args, kwargs)
            runtime_model = kwargs.pop("model", kwargs.pop("_model", None))

            run_id = None
            if self.tracer:
                run_id = self.tracer.start_run(
                    name=self.agent_name, run_type="agent", inputs=input_context, extra={"runtime_model": runtime_model}
                )

            try:
                if inspect.iscoroutinefunction(method_ref):

                    async def async_inner():
                        result = await self._execute_async(input_context, return_type, runtime_model)
                        if self.tracer and run_id:
                            self.tracer.end_run(run_id, outputs=result)
                        return result

                    return async_inner()
                else:
                    result = self._execute(input_context, return_type, runtime_model)
                    if self.tracer and run_id:
                        self.tracer.end_run(run_id, outputs=result)
                    return result

            except Exception as e:
                if self.tracer and run_id:
                    self.tracer.end_run(run_id, error=e)
                raise

        return method_wrapper

    def _extract_input_context(self, sig: inspect.Signature, args: tuple, kwargs: dict) -> Dict[str, Any]:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        context = {}
        for name, val in bound.arguments.items():
            if name in ("model", "_model", "self"):
                continue
            context[name] = str(val)
        return context

    async def _execute_async(
        self, input_context: Dict[str, Any], return_type: Type, runtime_model: Optional[str]
    ) -> Any:
        return await asyncio.to_thread(self._execute, input_context, return_type, runtime_model)

    def _execute(self, input_context: Dict[str, Any], return_type: Type, runtime_model: Optional[str]) -> Any:
        config = self.config_service.get_config(self.agent_name)

        if not config.enabled:
            raise AgentDisabledError(self.agent_name)

        final_model_name = self.model_router.resolve_model(capability=config.capability, runtime_override=runtime_model)

        llm = self.llm_factory.create(
            model_name=final_model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            llm_profile=config.llm_profile,
        )

        resolved_tools = self._resolve_dependencies(config)
        messages = build_messages(config, input_context)

        target_schema = return_type if self._is_pydantic_model(return_type) else None

        if config.agent_type == AgentType.REACT:
            return llm.invoke_agent_loop(messages, resolved_tools, config.max_iterations, output_schema=target_schema)
        else:
            if target_schema:
                return llm.invoke_structured(messages, resolved_tools, target_schema)
            return llm.invoke(messages, resolved_tools)

    def _resolve_dependencies(self, config: AgentConfig) -> List[Any]:
        """Resolve all tool dependencies for agent execution."""
        final_tools = []

        # Resolve named tools
        for tool_name in config.tools:
            tool = self._resolve_tool(tool_name)
            if tool:
                final_tools.append(self._wrap_tool(tool))

        # Resolve child agents as tools
        if self.locator:
            self._resolve_child_agents(config.agents, final_tools)

        # Add dynamic tools by tags
        self._add_dynamic_tools(config.tags, final_tools)

        return final_tools

    def _resolve_tool(self, tool_name: str) -> Any:
        """Resolve a tool by name from container or registry."""
        if self.container.has(tool_name):
            return self.container.get(tool_name)

        tool_ref = self.tool_registry.get_tool(tool_name)
        if tool_ref:
            return tool_ref() if isinstance(tool_ref, type) else tool_ref
        return None

    def _wrap_tool(self, tool_instance: Any) -> Any:
        """Wrap tool instance if needed."""
        if self._is_langchain_tool(tool_instance):
            return tool_instance
        if hasattr(type(tool_instance), TOOL_META_KEY):
            return ToolWrapper(tool_instance, getattr(type(tool_instance), TOOL_META_KEY))
        return tool_instance

    def _is_langchain_tool(self, obj: Any) -> bool:
        """Check if object is a LangChain-compatible tool."""
        return hasattr(obj, "args_schema") and hasattr(obj, "name") and hasattr(obj, "description")

    def _resolve_child_agents(self, agent_names: List[str], final_tools: List[Any]) -> None:
        """Resolve child agents and add them as tools."""
        for agent_name in agent_names:
            try:
                adapter = self._create_agent_tool(agent_name)
                if adapter:
                    final_tools.append(adapter)
            except AgentDisabledError:
                logger.debug("Child agent disabled, skipping: %s", agent_name)
            except ValueError as e:
                logger.warning("Failed to resolve child agent %s: %s", agent_name, e)

    def _create_agent_tool(self, agent_name: str) -> Optional[AgentAsTool]:
        """Create an AgentAsTool for a child agent."""
        child_agent = self.locator.get_agent(agent_name)
        if not child_agent:
            return None

        child_config = self.config_service.get_config(agent_name)
        if not child_config.enabled:
            return None

        method_name = self._get_agent_method_name(child_agent)
        return AgentAsTool(child_agent, method_name)

    def _get_agent_method_name(self, agent: Any) -> str:
        """Determine the method name to use for agent invocation."""
        if not agent.protocol_cls:
            return "invoke"

        candidates = [
            n
            for n, m in inspect.getmembers(agent.protocol_cls)
            if not n.startswith("_") and (inspect.isfunction(m) or inspect.ismethod(m))
        ]
        if "invoke" in candidates:
            return "invoke"
        return candidates[0] if candidates else "invoke"

    def _add_dynamic_tools(self, tags: List[str], final_tools: List[Any]) -> None:
        """Add dynamic tools by tags, avoiding duplicates."""
        for dt in self.tool_registry.get_dynamic_tools(tags):
            if dt not in final_tools:
                final_tools.append(dt)

    def _is_pydantic_model(self, cls: Type) -> bool:
        try:
            return issubclass(cls, BaseModel)
        except TypeError:
            return False
