from typing import Any, Callable

from pico_ioc import MethodCtx, MethodInterceptor, component

from .decorators import AGENT_META_KEY
from .proxy import TracedAgentProxy


@component
class AgentInterceptor(MethodInterceptor):
    def __init__(self, proxy: TracedAgentProxy):
        self.proxy = proxy

    def invoke(self, ctx: MethodCtx, call_next: Callable[[MethodCtx], Any]) -> Any:
        config = getattr(ctx.cls, AGENT_META_KEY, None)

        if not config or ctx.name != "invoke":
            return call_next(ctx)

        user_input = ""
        if ctx.args:
            user_input = str(ctx.args[0])
        elif "input" in ctx.kwargs:
            user_input = str(ctx.kwargs["input"])
        elif "message" in ctx.kwargs:
            user_input = str(ctx.kwargs["message"])

        return self.proxy.execute_agent(config.name, user_input)
