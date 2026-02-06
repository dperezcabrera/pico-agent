# Architecture

## Overview

Pico-Agent is built on top of Pico-IoC, leveraging its dependency injection capabilities to create a flexible and extensible multi-agent framework.

## Core Components

### Agent Proxy

The `AgentProxy` handles agent lifecycle and execution:

```
┌─────────────────────────────────────────────┐
│                AgentProxy                    │
├─────────────────────────────────────────────┤
│ - Manages agent configuration               │
│ - Handles tool registration                 │
│ - Coordinates with LLM provider             │
│ - Manages child agents                      │
└─────────────────────────────────────────────┘
```

### Tool System

Tools are injectable components that extend agent capabilities:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    Tool     │────▶│  ToolProxy  │────▶│    Agent    │
└─────────────┘     └─────────────┘     └─────────────┘
       │
       ▼
┌─────────────┐
│  Services   │
│ (injected)  │
└─────────────┘
```

### LLM Providers

Provider abstraction for multiple LLM backends:

```
┌─────────────────────────────────────────────┐
│              LLMFactory                      │
├─────────────────────────────────────────────┤
│ ┌─────────┐ ┌─────────┐ ┌─────────┐        │
│ │ OpenAI  │ │Anthropic│ │ Google  │  ...   │
│ └─────────┘ └─────────┘ └─────────┘        │
└─────────────────────────────────────────────┘
```

## Agent Execution Flow

```
1. User Request
       │
       ▼
2. Agent receives message
       │
       ▼
3. Agent sends to LLM with tools
       │
       ▼
4. LLM responds (may call tools)
       │
       ├──▶ Tool Call ──▶ Execute Tool ──▶ Return to LLM
       │
       ▼
5. Final response returned
```

## Multi-Agent Orchestration

```
┌─────────────────────────────────────────────┐
│           Orchestrator Agent                 │
├─────────────────────────────────────────────┤
│                    │                         │
│    ┌───────────────┼───────────────┐        │
│    ▼               ▼               ▼        │
│ ┌──────┐      ┌──────┐      ┌──────┐       │
│ │Agent1│      │Agent2│      │Agent3│       │
│ └──────┘      └──────┘      └──────┘       │
└─────────────────────────────────────────────┘
```

## Tracing

Built-in tracing captures:

- Agent invocations
- Tool calls
- LLM requests/responses
- Timing information
- Token usage

## Configuration

Configuration hierarchy:

```
Global Config
    │
    ├── Provider Config
    │       └── API keys, endpoints
    │
    ├── Agent Config
    │       └── Model, temperature, tools
    │
    └── Tracing Config
            └── Exporters, sampling
```
