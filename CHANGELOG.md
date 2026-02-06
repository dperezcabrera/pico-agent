# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.html).

---

## [0.2.0] - 2026-02-06

### Added
- **Lifecycle Management**: `AgentSystem` singleton component that coordinates container lifecycle phases (INITIALIZING → SCANNING → READY → RUNNING → SHUTTING_DOWN → STOPPED).
- **Lifecycle Events**: `LifecycleEvent` dataclass published via pico-ioc `EventBus` on every phase transition.
- **Standalone Bootstrap**: `pico_agent.init()` wraps `pico_ioc.init()` adding automatic module inclusion and plugin discovery — pico-agent can now boot independently without pico-boot.
- **Plugin System**: `pico_agent.plugins` entry-points group for third-party plugin auto-discovery, controlled via `PICO_AGENT_AUTO_PLUGINS` env var.
- **`AgentLifecycleError`** exception class for lifecycle-related failures.
- **TraceService cleanup**: Automatic trace flushing on container shutdown via `@cleanup` hook.
- `tests/test_lifecycle.py`: 10 tests covering phases, events, transitions, and EventBus integration.
- `tests/test_bootstrap.py`: 19 tests covering helpers, normalization, scanner harvesting, and init behavior.

### Changed
- `__init__.py`: Exports `AgentSystem`, `LifecyclePhase`, `LifecycleEvent`, `AgentLifecycleError`, and `init`.
- `pyproject.toml`: Added `pico_agent.plugins` entry-points group.

---

## [0.1.1] - 2025-02-04

### Changed
- **Code Quality**: Major refactoring to reduce cyclomatic complexity from D(25) to A(2.5).
  - `proxy.py`: Refactored `_resolve_dependencies` method by extracting 7 helper methods:
    - `_resolve_tool`, `_wrap_tool`, `_is_langchain_tool`
    - `_resolve_child_agents`, `_create_agent_tool`, `_get_agent_method_name`
    - `_add_dynamic_tools`
  - `providers.py`: Refactored `create_chat_model` from C(15) to A(2) using dict dispatch pattern:
    - Extracted `_require_key` helper
    - Created individual provider methods: `_create_openai`, `_create_azure`, `_create_gemini`, `_create_anthropic`, `_create_deepseek`, `_create_qwen`
- **Documentation**: Standardized MkDocs configuration with Material theme (indigo), git-revision-date-localized plugin, and math extensions.
- **CI/CD**: Unified GitHub Actions workflow for documentation deployment.

### Fixed
- **Test Coverage**: Improved overall test coverage from 82% to 90%.
  - `providers.py`: Increased from 44% to 82%.
  - `locator.py`: Increased from 77% to 98%.

### Added
- `tests/test_providers_coverage.py`: 38 new tests for `LangChainAdapter` and `LangChainLLMFactory`.
- `tests/test_locator_coverage.py`: 11 new tests for `AgentLocator` and `NoOpCentralClient`.
- `src/pico_agent/logging.py`: Centralized logging module with `get_logger` and `configure_logging`.

---

## [0.1.0] - 2025-01-15

### Added
- Initial release of `pico-agent`.
- **Agent Framework**: Protocol-based agent definitions with `@agent` decorator.
- **Multi-Provider Support**: OpenAI, Anthropic, Google Gemini, Azure, DeepSeek, Qwen.
- **Tool Integration**: LangChain-compatible tools and dynamic tool registration.
- **Agent Composition**: Hierarchical agent orchestration with child agents as tools.
- **Tracing**: Built-in tracing support with LangSmith integration.
- **Virtual Agents**: YAML-based agent configuration without code.
- **Model Router**: Capability-based model selection and runtime overrides.
- **Experiment Registry**: A/B testing support for agent variants.
