# Architecture

## Overview

Pico-Agent is built on top of Pico-IoC, leveraging its dependency injection capabilities to create a flexible and extensible multi-agent framework.

## Agent Execution Flow

The following diagram shows the complete lifecycle of an agent invocation, from user request to final response:

```mermaid
flowchart TD
    A["User calls agent.method()"] --> B["DynamicAgentProxy.__getattr__()"]
    B --> C["Extract input context from method args"]
    C --> D{"TraceService available?"}
    D -- Yes --> E["tracer.start_run()"]
    D -- No --> F["Skip tracing"]
    E --> F
    F --> G["AgentConfigService.get_config()"]
    G --> H{"Agent enabled?"}
    H -- No --> I["Raise AgentDisabledError"]
    H -- Yes --> J["ModelRouter.resolve_model()"]
    J --> K["LLMFactory.create()"]
    K --> L["Resolve tools and child agents"]
    L --> M["build_messages()"]
    M --> N{"Agent type?"}
    N -- ONE_SHOT --> O["LLM.invoke()"]
    N -- REACT --> P["LLM.invoke_agent_loop()"]
    N -- WORKFLOW --> Q["VirtualAgentRunner._arun_workflow()"]
    O --> R{"Return type is Pydantic?"}
    R -- Yes --> S["LLM.invoke_structured()"]
    R -- No --> T["Return text response"]
    P --> T
    Q --> T
    S --> T
    T --> U["tracer.end_run()"]
    U --> V["Return result to caller"]
```

## Provider Resolution

This diagram shows how pico-agent selects the correct LLM provider and creates the appropriate chat model:

```mermaid
flowchart TD
    A["Agent needs LLM"] --> B["ModelRouter.resolve_model()"]
    B --> C{"_model runtime override?"}
    C -- Yes --> D["Use override model name"]
    C -- No --> E["Map capability to default model"]
    D --> F{"Contains ':' separator?"}
    E --> F
    F -- Yes --> G["Split 'provider:model'"]
    F -- No --> H["Auto-detect provider"]
    G --> I["LangChainLLMFactory.create_chat_model()"]
    H --> I

    I --> J{"Provider?"}
    J -- openai --> K["ChatOpenAI"]
    J -- azure --> L["AzureChatOpenAI"]
    J -- gemini/google --> M["ChatGoogleGenerativeAI"]
    J -- claude/anthropic --> N["ChatAnthropic"]
    J -- deepseek --> O["ChatOpenAI + DeepSeek URL"]
    J -- qwen --> P["ChatOpenAI + Qwen URL"]
    J -- unknown --> Q["Raise ValueError"]

    K --> R["Apply temperature & max_tokens"]
    L --> R
    M --> R
    N --> R
    O --> R
    P --> R
    R --> S["Wrap in LangChainAdapter"]
    S --> T["Return LLM instance"]
```

## Tool Registration and Invocation

This diagram shows the two paths for registering tools and how they are resolved at agent execution time:

```mermaid
flowchart TD
    subgraph Registration
        A["@tool(name, description)"] --> B["ToolScanner.auto_scan()"]
        B --> C["ToolScanner.scan_module()"]
        C --> D["ToolRegistry.register()"]

        E["VirtualToolManager.create_tool()"] --> F["DynamicTool()"]
        F --> D
    end

    subgraph Resolution["Tool Resolution at Execution"]
        G["DynamicAgentProxy._resolve_dependencies()"] --> H["Resolve named tools from config.tools"]
        H --> I{"Tool in container?"}
        I -- Yes --> J["container.get(tool_name)"]
        I -- No --> K["ToolRegistry.get_tool()"]
        J --> L{"Is LangChain tool?"}
        K --> L
        L -- Yes --> M["Use as-is"]
        L -- No --> N{"Has TOOL_META_KEY?"}
        N -- Yes --> O["Wrap with ToolWrapper"]
        N -- No --> P["Use as-is"]

        G --> Q["Resolve child agents from config.agents"]
        Q --> R["AgentLocator.get_agent()"]
        R --> S["Wrap as AgentAsTool"]

        G --> T["ToolRegistry.get_dynamic_tools(tags)"]
        T --> U["Add tag-matched + global tools"]
    end

    M --> V["All tools bound to LLM"]
    O --> V
    P --> V
    S --> V
    U --> V
```

## Configuration Hierarchy

```mermaid
flowchart TD
    A["AgentConfigService.get_config(name)"] --> B["CentralConfigClient.get_agent_config()"]
    A --> C["LocalAgentRegistry.get_config()"]
    A --> D["Runtime overrides"]
    B --> E{"Remote config exists?"}
    E -- Yes --> F["Use remote as base"]
    E -- No --> G{"Local config exists?"}
    G -- Yes --> H["Use local as base"]
    G -- No --> I{"Runtime overrides exist?"}
    I -- Yes --> J["Create AgentConfig from runtime data"]
    I -- No --> K["Raise ValueError"]
    F --> L{"Runtime overrides?"}
    H --> L
    L -- Yes --> M["Merge: replace(base, **runtime)"]
    L -- No --> N["Return base config"]
    M --> O["Return merged config"]
    J --> O
```

## Multi-Agent Orchestration

```mermaid
flowchart TD
    A["Orchestrator Agent<br/>AgentType.REACT"] --> B["LLM decides to call child agent"]
    B --> C["AgentAsTool.__call__()"]
    C --> D["Child DynamicAgentProxy.method()"]
    D --> E["Child agent's own LLM call"]
    E --> F["Result returned to orchestrator"]
    F --> G{"More tool calls needed?"}
    G -- Yes --> B
    G -- No --> H["Final response"]
```

## Map-Reduce Workflow

```mermaid
flowchart TD
    A["VirtualAgentRunner.arun()"] --> B["Splitter Agent"]
    B --> C["SplitterOutput: List of TaskItem"]
    C --> D["Distribute tasks via LangGraph Send"]
    D --> E1["Mapper Agent 1"]
    D --> E2["Mapper Agent 2"]
    D --> E3["Mapper Agent N"]
    E1 --> F["Collect mapped_results"]
    E2 --> F
    E3 --> F
    F --> G["Reducer Agent"]
    G --> H["Final combined output"]
```

## Component Relationships

```mermaid
classDiagram
    class AgentLocator {
        +get_agent(name_or_protocol)
        +create_proxy(protocol)
    }
    class DynamicAgentProxy {
        +__getattr__(name)
    }
    class VirtualAgentRunner {
        +run(input)
        +arun(input)
        +run_structured(input, schema)
    }
    class AgentConfigService {
        +get_config(name)
        +update_agent_config(name, **kwargs)
    }
    class ToolRegistry {
        +register(name, tool, tags)
        +get_tool(name)
        +get_dynamic_tools(tags)
    }
    class ModelRouter {
        +resolve_model(capability, override)
        +update_mapping(capability, model)
    }
    class LangChainLLMFactory {
        +create(model_name, temperature, max_tokens)
    }
    class LangChainAdapter {
        +invoke(messages, tools)
        +invoke_structured(messages, tools, schema)
        +invoke_agent_loop(messages, tools, max_iter)
    }
    class TraceService {
        +start_run(name, type, inputs)
        +end_run(run_id, outputs, error)
    }

    AgentLocator --> DynamicAgentProxy : creates
    AgentLocator --> VirtualAgentRunner : creates
    AgentLocator --> AgentConfigService : uses
    AgentLocator --> ModelRouter : uses
    DynamicAgentProxy --> AgentConfigService : uses
    DynamicAgentProxy --> ToolRegistry : uses
    DynamicAgentProxy --> LangChainLLMFactory : uses
    DynamicAgentProxy --> ModelRouter : uses
    DynamicAgentProxy --> TraceService : uses
    LangChainLLMFactory --> LangChainAdapter : creates
    VirtualAgentRunner --> LangChainLLMFactory : uses
    VirtualAgentRunner --> ToolRegistry : uses
```

## Tracing

Built-in tracing captures:

- Agent invocations
- Tool calls
- LLM requests/responses
- Timing information
- Parent-child relationships via `run_context` ContextVar

## Configuration

Configuration hierarchy:

```
Global Config
    |
    |-- Provider Config
    |       +-- API keys, endpoints (LLMConfig)
    |
    |-- Agent Config
    |       +-- Model capability, temperature, tools
    |
    +-- Tracing Config
            +-- TraceService singleton
```
