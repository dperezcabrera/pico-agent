import pytest
from pico_agent.router import ModelRouter
from pico_agent.config import AgentCapability


class TestModelRouter:
    @pytest.fixture
    def router(self):
        return ModelRouter()

    def test_resolve_fast_capability(self, router):
        model = router.resolve_model(AgentCapability.FAST)
        assert model == "gpt-5-mini"

    def test_resolve_smart_capability(self, router):
        model = router.resolve_model(AgentCapability.SMART)
        assert model == "gpt-5.1"

    def test_resolve_reasoning_capability(self, router):
        model = router.resolve_model(AgentCapability.REASONING)
        assert model == "gemini-3-pro"

    def test_resolve_vision_capability(self, router):
        model = router.resolve_model(AgentCapability.VISION)
        assert model == "gpt-4o"

    def test_resolve_coding_capability(self, router):
        model = router.resolve_model(AgentCapability.CODING)
        assert model == "claude-3-5-sonnet"

    def test_resolve_unknown_capability_returns_default(self, router):
        model = router.resolve_model("unknown_capability")
        assert model == "gpt-5.1"

    def test_runtime_override_takes_precedence(self, router):
        model = router.resolve_model(
            AgentCapability.FAST,
            runtime_override="claude-opus-4"
        )
        assert model == "claude-opus-4"

    def test_update_mapping(self, router):
        router.update_mapping(AgentCapability.FAST, "new-fast-model")
        model = router.resolve_model(AgentCapability.FAST)
        assert model == "new-fast-model"

    def test_update_mapping_for_custom_capability(self, router):
        router.update_mapping("custom", "custom-model")
        model = router.resolve_model("custom")
        assert model == "custom-model"


class TestModelRouterIntegration:
    def test_router_with_all_capabilities(self):
        router = ModelRouter()

        capabilities = [
            AgentCapability.FAST,
            AgentCapability.SMART,
            AgentCapability.REASONING,
            AgentCapability.VISION,
            AgentCapability.CODING
        ]

        for cap in capabilities:
            model = router.resolve_model(cap)
            assert model is not None
            assert len(model) > 0

    def test_router_isolation(self):
        router1 = ModelRouter()
        router2 = ModelRouter()

        router1.update_mapping(AgentCapability.FAST, "modified-model")

        assert router1.resolve_model(AgentCapability.FAST) == "modified-model"
        assert router2.resolve_model(AgentCapability.FAST) == "gpt-5-mini"
