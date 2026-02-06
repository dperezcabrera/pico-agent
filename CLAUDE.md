Read and follow ./AGENTS.md for project conventions.

## Pico Ecosystem Context

pico-agent provides LLM agent orchestration for pico-ioc. It uses:
- `@component` for scanners, registries, router, locator, tracing, scheduler
- `@factory` + `@provides` for CentralConfigClient, LLMConfig, LLMFactory
- `@configure` hooks for AgentScanner and ToolScanner auto-discovery
- `MethodInterceptor` for AgentInterceptor
- AgentScanner inherits from DeferredProvider (custom scanner)
- Auto-discovered via `pico_boot.modules` entry point

## Key Reminders

- pico-ioc dependency: `>= 2.2.0`
- `version_scheme = "guess-next-dev"` (clean versions on tag)
- requires-python >= 3.11
- Commit messages: one line only
- LLM providers: OpenAI, Azure, Google/Gemini, Anthropic/Claude, DeepSeek, Qwen
- AgentCapability defaults map to specific models (e.g., SMART → "gpt-5.1")
- `run_context` ContextVar is for trace hierarchy, NOT for DI scoping
