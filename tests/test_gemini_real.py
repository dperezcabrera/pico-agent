import os
import pytest
from typing import Protocol
from pico_ioc import init, component, configure
from pico_agent import agent, AgentCapability, AgentType
from pico_agent.config import LLMConfig
from pico_agent.scanner import AgentScanner
from pico_agent.router import ModelRouter

@agent(
    name="gemini_translator",
    capability=AgentCapability.FAST,
    system_prompt="You are a professional translator. Translate the input to Spanish.",
    user_prompt_template="{text}",
    agent_type=AgentType.ONE_SHOT,
    temperature=0.1,
)
class GeminiTranslator(Protocol):
    def translate(self, text: str) -> str: ...

@component
class TestConfiguration:
    @configure
    def setup_llm_keys(self, config: LLMConfig):
        key = os.getenv("GOOGLE_API_KEY")
        config.api_keys["google"] = key

@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"), 
    reason="GOOGLE_API_KEY not found in environment variables"
)
def test_real_gemini_translation_with_configure():
    container = init(
        modules=["pico_agent", __name__],
        custom_scanners=[AgentScanner()]
    )

    container.get(TestConfiguration)

    router = container.get(ModelRouter)
    router.update_mapping(AgentCapability.FAST, "gemini:gemini-2.5-flash")

    translator = container.get(GeminiTranslator)
    
    result = translator.translate(text="Hello world, this is a test using @configure and llm_profile.")

    assert result is not None
    assert len(result) > 0
    assert "Hola" in result or "prueba" in result.lower()
