"""Agent system lifecycle management.

``AgentSystem`` tracks the framework's lifecycle phases and publishes
``LifecycleEvent`` notifications via pico-ioc's ``EventBus``.
"""

from dataclasses import dataclass
from enum import Enum

from pico_ioc import Event, EventBus, PicoContainer, cleanup, component, configure

from .logging import get_logger

logger = get_logger(__name__)


class LifecyclePhase(str, Enum):
    """Phases of the pico-agent system lifecycle.

    Attributes:
        INITIALIZING: Container is being built.
        SCANNING: Agents and tools are being discovered.
        READY: Container is fully configured.
        RUNNING: System is accepting requests.
        SHUTTING_DOWN: Graceful shutdown in progress.
        STOPPED: System has stopped.
    """

    INITIALIZING = "initializing"
    SCANNING = "scanning"
    READY = "ready"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


@dataclass
class LifecycleEvent(Event):
    """Event published when the system transitions between lifecycle phases.

    Args:
        phase: The new ``LifecyclePhase``.
        detail: Optional human-readable detail string.
    """

    phase: LifecyclePhase
    detail: str = ""


@component(scope="singleton")
class AgentSystem:
    """Lifecycle coordinator that publishes phase transitions via ``EventBus``.

    Transitions are published as ``LifecycleEvent`` instances.  The system
    moves through: ``INITIALIZING`` -> ``READY`` -> ``RUNNING`` ->
    ``SHUTTING_DOWN`` -> ``STOPPED``.
    """

    def __init__(self):
        self._phase = LifecyclePhase.INITIALIZING
        self._event_bus: EventBus | None = None

    @property
    def phase(self) -> LifecyclePhase:
        """The current lifecycle phase."""
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
