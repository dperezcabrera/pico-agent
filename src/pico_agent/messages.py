from typing import Any, Dict, List
from .config import AgentConfig


def build_messages(config: AgentConfig, context: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build LLM message list from agent config and input context."""
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
