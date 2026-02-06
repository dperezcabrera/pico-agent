from enum import Enum
from dataclasses import dataclass

from pico_ioc import component, configure, cleanup, PicoContainer
from pico_ioc import Event, EventBus

from .logging import get_logger

logger = get_logger(__name__)


class LifecyclePhase(str, Enum):
    INITIALIZING = "initializing"
    SCANNING = "scanning"
    READY = "ready"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


@dataclass
class LifecycleEvent(Event):
    phase: LifecyclePhase
    detail: str = ""


@component(scope="singleton")
class AgentSystem:
    """Lifecycle coordinator — publishes transitions via EventBus de pico-ioc."""

    def __init__(self):
        self._phase = LifecyclePhase.INITIALIZING
        self._event_bus: EventBus | None = None

    @property
    def phase(self) -> LifecyclePhase:
        return self._phase

    def _transition(self, phase: LifecyclePhase, detail: str = ""):
        self._phase = phase
        if self._event_bus:
            self._event_bus.publish_sync(LifecycleEvent(phase=phase, detail=detail))

    @configure
    def _on_ready(self, container: PicoContainer):
        if container.has(EventBus):
            self._event_bus = container.get(EventBus)
        self._transition(LifecyclePhase.READY, "Container configured")
        self._transition(LifecyclePhase.RUNNING)

    @cleanup
    def _on_shutdown(self):
        self._transition(LifecyclePhase.SHUTTING_DOWN)
        self._transition(LifecyclePhase.STOPPED)
