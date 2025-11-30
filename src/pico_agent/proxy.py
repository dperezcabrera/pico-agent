import inspect
from typing import Any, List, Dict, Type, get_type_hints, Optional
from pydantic import BaseModel
from pico_ioc import component, PicoContainer
from .config import AgentConfig, AgentType
from .registry import AgentConfigService, ToolRegistry
from .interfaces import LLMFactory
from .router import ModelRouter
from .exceptions import AgentDisabledError
from .tools import AgentAsTool, ToolWrapper
from .decorators import TOOL_META_KEY
from .tracing import TraceService

@component
class TracedAgentProxy:
    def __init__(
        self,
        config_service: AgentConfigService,
        tool_registry: ToolRegistry,
        llm_factory: LLMFactory,
        model_router: ModelRouter
    ):
        self.config_service = config_service
        self.tool_registry = tool_registry
        self.llm_factory = llm_factory
        self.model_router = model_router

    def execute_agent(self, agent_name: str, user_input: str) -> Any:
        config = self.config_service.get_config(agent_name)
        
        if not config.enabled:
            raise AgentDisabledError(agent_name)
        
        final_model_name = self.model_router.resolve_model(
            capability=config.capability,
            runtime_override=None
        )
        
        llm = self.llm_factory.create(
            model_name=final_model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens
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
            return llm.invoke_agent_loop(
                messages, 
                resolved_tools, 
                config.max_iterations
            )
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
        locator: Any = None 
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
            new_params = params[1:]
            method_sig = method_sig.replace(parameters=new_params)

        type_hints = get_type_hints(method_ref)
        return_type = type_hints.get("return", str)

        def method_wrapper(*args, **kwargs):
            input_context = self._extract_input_context(method_sig, args, kwargs)
            runtime_model = kwargs.pop("model", kwargs.pop("_model", None))
            
            run_id = None
            if self.tracer:
                run_id = self.tracer.start_run(
                    name=self.agent_name,
                    run_type="agent",
                    inputs=input_context,
                    extra={"runtime_model": runtime_model}
                )

            try:
                result = self._execute(input_context, return_type, runtime_model)
                if self.tracer and run_id:
                    self.tracer.end_run(run_id, outputs=result)
                return result
            except Exception as e:
                if self.tracer and run_id:
                    self.tracer.end_run(run_id, error=e)
                raise e
        
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

    def _execute(self, input_context: Dict[str, Any], return_type: Type, runtime_model: Optional[str]) -> Any:
        config = self.config_service.get_config(self.agent_name)
        
        if not config.enabled:
            raise AgentDisabledError(self.agent_name)
        
        final_model_name = self.model_router.resolve_model(
            capability=config.capability,
            runtime_override=runtime_model
        )
        
        llm = self.llm_factory.create(
            model_name=final_model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            llm_profile=config.llm_profile
        )

        resolved_tools = self._resolve_dependencies(config)
        messages = self._build_messages(config, input_context)
        
        target_schema = return_type if self._is_pydantic_model(return_type) else None

        if config.agent_type == AgentType.REACT:
            return llm.invoke_agent_loop(
                messages, 
                resolved_tools, 
                config.max_iterations, 
                output_schema=target_schema
            )
        else:
            if target_schema:
                return llm.invoke_structured(messages, resolved_tools, target_schema)
            return llm.invoke(messages, resolved_tools)

    def _resolve_dependencies(self, config: AgentConfig) -> List[Any]:
        final_tools = []
        
        for tool_name in config.tools:
            tool_instance = None
            
            if self.container.has(tool_name):
                tool_instance = self.container.get(tool_name)
            
            elif self.tool_registry.get_tool(tool_name):
                tool_ref = self.tool_registry.get_tool(tool_name)
                if isinstance(tool_ref, type):
                    tool_instance = tool_ref()
                else:
                    tool_instance = tool_ref
            
            if tool_instance:
                if hasattr(tool_instance, "args_schema") and hasattr(tool_instance, "name") and hasattr(tool_instance, "description"):
                    final_tools.append(tool_instance)
                elif hasattr(type(tool_instance), TOOL_META_KEY):
                    tool_config = getattr(type(tool_instance), TOOL_META_KEY)
                    final_tools.append(ToolWrapper(tool_instance, tool_config))
                else:
                    final_tools.append(tool_instance)

        if self.locator:
            for agent_name in config.agents:
                try:
                    child_agent = self.locator.get_agent(agent_name)
                    if child_agent:
                        child_config = self.config_service.get_config(agent_name)
                        if not child_config.enabled: continue
                        
                        method_name = "invoke"
                        if child_agent.protocol_cls:
                            protocol = child_agent.protocol_cls
                            candidates = [n for n, m in inspect.getmembers(protocol) if not n.startswith("_") and (inspect.isfunction(m) or inspect.ismethod(m))]
                            if "invoke" in candidates:
                                method_name = "invoke"
                            elif candidates:
                                method_name = candidates[0]
                        
                        adapter = AgentAsTool(child_agent, method_name)
                        final_tools.append(adapter)
                except Exception:
                    pass

        dynamic = self.tool_registry.get_dynamic_tools(config.tags)
        for dt in dynamic:
            if dt not in final_tools:
                final_tools.append(dt)
        
        return final_tools

    def _build_messages(self, config: AgentConfig, input_context: Dict[str, Any]) -> List[Dict[str, str]]:
        messages = []
        if config.system_prompt:
            try:
                sys_content = config.system_prompt.format(**input_context)
            except KeyError:
                sys_content = config.system_prompt
            messages.append({"role": "system", "content": sys_content})
        
        user_content = " ".join(input_context.values())
        if config.user_prompt_template:
            try:
                user_content = config.user_prompt_template.format(**input_context)
            except KeyError:
                pass
        
        messages.append({"role": "user", "content": user_content})
        return messages

    def _is_pydantic_model(self, cls: Type) -> bool:
        try:
            return issubclass(cls, BaseModel)
        except TypeError:
            return False
