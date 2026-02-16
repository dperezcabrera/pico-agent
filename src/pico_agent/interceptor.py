"""Method interceptor for automatic agent execution.

``AgentInterceptor`` is a pico-ioc ``MethodInterceptor`` that intercepts
calls to the ``invoke`` method on ``@agent``-decorated classes and routes
them through ``TracedAgentProxy``.
"""

from typing import Any, Callable

from pico_ioc import MethodCtx, MethodInterceptor, component

from .decorators import AGENT_META_KEY
from .proxy import TracedAgentProxy


@component
class AgentInterceptor(MethodInterceptor):
    """Intercepts ``invoke`` calls on agent classes and delegates to the LLM.

    Only intercepts methods named ``"invoke"`` on classes that carry
    ``AGENT_META_KEY`` metadata.  All other calls are passed through
    unchanged.

    Args:
        proxy: The ``TracedAgentProxy`` used for agent execution.
    """

    def __init__(self, proxy: TracedAgentProxy):
        self.proxy = proxy

    def invoke(self, ctx: MethodCtx, call_next: Callable[[MethodCtx], Any]) -> Any:
        """Intercept an agent method call.

        If the target class has agent metadata and the method is ``"invoke"``,
        the call is routed to ``TracedAgentProxy.execute_agent()``.
        Otherwise, the call proceeds normally via *call_next*.

        Args:
            ctx: The method invocation context.
            call_next: Callable to proceed with the original method.

        Returns:
            The agent's LLM response, or the original method's return value.
        """
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
