from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum

class AgentType(str, Enum):
    ONE_SHOT = "one_shot"
    REACT = "react"

class AgentCapability:
    FAST = "fast"
    SMART = "smart"
    REASONING = "reasoning"
    VISION = "vision"
    CODING = "coding"

@dataclass
class AgentConfig:
    name: str
    system_prompt: str
    user_prompt_template: str = "{input}"
    description: str = ""
    capability: str = AgentCapability.SMART
    enabled: bool = True
    agent_type: AgentType = AgentType.ONE_SHOT
    max_iterations: int = 5
    tools: List[str] = field(default_factory=list)
    agents: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    tracing_enabled: bool = True
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    llm_profile: Optional[str] = None

@dataclass
class ToolConfig:
    name: str
    description: str

@dataclass
class LLMConfig:
    api_keys: Dict[str, str] = field(default_factory=dict)
    base_urls: Dict[str, str] = field(default_factory=dict)
