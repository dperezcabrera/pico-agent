import pytest
from unittest.mock import MagicMock, call

from pico_ioc import init, EventBus, Event
from pico_agent.lifecycle import AgentSystem, LifecyclePhase, LifecycleEvent


class TestLifecyclePhase:
    def test_all_phases_exist(self):
        phases = [
            LifecyclePhase.INITIALIZING,
            LifecyclePhase.SCANNING,
            LifecyclePhase.READY,
            LifecyclePhase.RUNNING,
            LifecyclePhase.SHUTTING_DOWN,
            LifecyclePhase.STOPPED,
        ]
        assert len(phases) == 6

    def test_phases_are_strings(self):
        assert LifecyclePhase.INITIALIZING == "initializing"
        assert LifecyclePhase.RUNNING == "running"
        assert LifecyclePhase.STOPPED == "stopped"


class TestLifecycleEvent:
    def test_event_creation(self):
        event = LifecycleEvent(phase=LifecyclePhase.READY, detail="test")
        assert event.phase == LifecyclePhase.READY
        assert event.detail == "test"

    def test_event_default_detail(self):
        event = LifecycleEvent(phase=LifecyclePhase.RUNNING)
        assert event.detail == ""

    def test_is_event_subclass(self):
        assert issubclass(LifecycleEvent, Event)


class TestAgentSystem:
    def test_initial_phase_is_initializing(self):
        system = AgentSystem()
        assert system.phase == LifecyclePhase.INITIALIZING

    def test_configure_transitions_to_running(self):
        container = init(modules=["pico_agent"])
        system = container.get(AgentSystem)
        assert system.phase == LifecyclePhase.RUNNING

    def test_shutdown_transitions_to_stopped(self):
        system = AgentSystem()
        mock_container = MagicMock()
        mock_container.has.return_value = False

        system._on_ready(mock_container)
        assert system.phase == LifecyclePhase.RUNNING

        system._on_shutdown()
        assert system.phase == LifecyclePhase.STOPPED

    def test_event_bus_receives_lifecycle_events(self):
        container = init(modules=["pico_ioc", "pico_agent"])
        bus = container.get(EventBus)
        system = container.get(AgentSystem)

        received = []
        bus.subscribe(LifecycleEvent, lambda e: received.append(e))

        system._on_shutdown()

        assert len(received) == 2
        assert received[0].phase == LifecyclePhase.SHUTTING_DOWN
        assert received[1].phase == LifecyclePhase.STOPPED

    def test_works_without_event_bus(self):
        system = AgentSystem()
        mock_container = MagicMock()
        mock_container.has.return_value = False

        system._on_ready(mock_container)
        assert system.phase == LifecyclePhase.RUNNING

        system._on_shutdown()
        assert system.phase == LifecyclePhase.STOPPED
