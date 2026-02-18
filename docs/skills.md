# AI Coding Skills

[Claude Code](https://code.claude.com) and [OpenAI Codex](https://openai.com/index/introducing-codex/) skills for AI-assisted development with pico-agent.

## Installation

```bash
curl -sL https://raw.githubusercontent.com/dperezcabrera/pico-skills/main/install.sh | bash -s -- agent
```

Or install all pico-framework skills:

```bash
curl -sL https://raw.githubusercontent.com/dperezcabrera/pico-skills/main/install.sh | bash
```

### Platform-specific

```bash
# Claude Code only
curl -sL https://raw.githubusercontent.com/dperezcabrera/pico-skills/main/install.sh | bash -s -- --claude agent

# OpenAI Codex only
curl -sL https://raw.githubusercontent.com/dperezcabrera/pico-skills/main/install.sh | bash -s -- --codex agent
```

## Available Commands

### `/add-agent`

Creates an LLM agent or tool with pico-agent. Use when building AI agents, chatbots, ReAct agents with tools, or one-shot LLM-powered components.

**Agent types:** ReAct agent (`@agent` with tools), one-shot agent (`@agent(agent_type="one_shot")`), virtual agent (YAML-defined), custom tool (`@tool`).

```
/add-agent support_bot
/add-agent summarizer --type one_shot
/add-agent search_tool --type tool
```

### `/add-component`

Creates a new pico-ioc component with dependency injection. Use when adding services, factories, or interceptors.

```
/add-component ToolService
```

### `/add-tests`

Generates tests for existing pico-framework components. Creates unit tests for agents and tools with mocked LLM responses.

```
/add-tests SupportAgent
/add-tests SearchTool
```

## More Information

See [pico-skills](https://github.com/dperezcabrera/pico-skills) for the full list of skills, selective installation, and details.
