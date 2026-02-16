"""Shared message builder for agent execution.

Converts ``AgentConfig`` system / user prompt templates and an input context
dictionary into the ``[{"role": ..., "content": ...}]`` message list expected
by the ``LLM`` protocol.
"""

from typing import Any, Dict, List

from .config import AgentConfig


def build_messages(config: AgentConfig, context: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build an LLM message list from agent config and input context.

    Template placeholders in ``config.system_prompt`` and
    ``config.user_prompt_template`` are filled using *context* keys.  If a
    placeholder cannot be resolved, the raw template is used as-is.

    Args:
        config: The agent's ``AgentConfig``.
        context: Mapping of parameter names to their string values, typically
            derived from the method signature of the invoked agent method.

    Returns:
        A list of message dicts with ``"role"`` and ``"content"`` keys,
        starting with an optional ``"system"`` message followed by a
        ``"user"`` message.
    """
    messages = []
    if config.system_prompt:
        try:
            sys_content = config.system_prompt.format(**context)
        except KeyError:
            sys_content = config.system_prompt
        messages.append({"role": "system", "content": sys_content})

    user_content = " ".join(str(v) for v in context.values())
    if config.user_prompt_template:
        try:
            user_content = config.user_prompt_template.format(**context)
        except KeyError:
            pass

    messages.append({"role": "user", "content": user_content})
    return messages
