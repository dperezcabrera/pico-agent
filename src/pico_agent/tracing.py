"""Observability tracing for agent, tool, and LLM invocations.

``TraceService`` collects hierarchical ``TraceRun`` records.  Parent-child
relationships are maintained automatically via the ``run_context``
``ContextVar``.  Traces are recorded by ``DynamicAgentProxy`` and
``LangChainAdapter``.
"""

import time
import uuid
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from pico_ioc import cleanup, component

from .logging import get_logger

logger = get_logger(__name__)

run_context: ContextVar[Optional[str]] = ContextVar("run_context", default=None)
"""ContextVar tracking the current trace run ID for parent-child hierarchy.

This is used for trace hierarchy only -- it is **not** used for DI scoping.
"""


@dataclass
class TraceRun:
    """A single trace record for an agent, tool, or LLM invocation.

    Attributes:
        id: Unique run identifier (UUID).
        name: Human-readable name (e.g., agent name or ``"LLM: gpt-5"``).
        run_type: Category string -- ``"agent"``, ``"llm"``, or ``"tool"``.
        inputs: Input data (e.g., messages, arguments).
        parent_id: ID of the parent run, or ``None`` for root runs.
        start_time: Unix timestamp when the run started.
        end_time: Unix timestamp when the run ended (set by ``end_run``).
        outputs: Output data (set by ``end_run``).
        error: Error message if the run failed (set by ``end_run``).
        extra: Arbitrary metadata (e.g., ``{"runtime_model": "gpt-4"}``).
    """

    id: str
    name: str
    run_type: str
    inputs: Dict[str, Any]
    parent_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    outputs: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@component(scope="singleton")
class TraceService:
    """Singleton service that collects hierarchical trace runs.

    Traces are stored in memory and can be retrieved via ``get_traces()``.
    On container shutdown (``@cleanup``), all traces are flushed.
    """

    def __init__(self):
        self.traces: List[TraceRun] = []

    def start_run(self, name: str, run_type: str, inputs: Dict[str, Any], extra: Dict[str, Any] = None) -> str:
        """Begin a new trace run.

        Automatically sets the parent ID from ``run_context`` and updates
        the context var to the new run ID.

        Args:
            name: Human-readable run name.
            run_type: Category (``"agent"``, ``"llm"``, ``"tool"``).
            inputs: Input data to record.
            extra: Optional metadata dict.

        Returns:
            The unique run ID (UUID string).
        """
        parent_id = run_context.get()
        run_id = str(uuid.uuid4())

        run = TraceRun(id=run_id, name=name, run_type=run_type, inputs=inputs, parent_id=parent_id, extra=extra or {})

        self.traces.append(run)
        run_context.set(run_id)
        return run_id

    def end_run(self, run_id: str, outputs: Any = None, error: Exception = None):
        """Complete a trace run, recording outputs or an error.

        Restores ``run_context`` to the parent run's ID.

        Args:
            run_id: The ID returned by ``start_run()``.
            outputs: Output data -- can be a string, dict, Pydantic model,
                or any object (converted via ``str()``).
            error: Exception instance if the run failed.
        """
        for run in reversed(self.traces):
            if run.id == run_id:
                run.end_time = time.time()
                if error:
                    run.error = str(error)
                else:
                    if isinstance(outputs, (str, int, float, bool)):
                        run.outputs = {"output": outputs}
                    elif hasattr(outputs, "dict"):
                        run.outputs = outputs.dict()
                    elif isinstance(outputs, dict):
                        run.outputs = outputs
                    else:
                        run.outputs = {"output": str(outputs)}

                run_context.set(run.parent_id)
                self._persist(run)
                break

    def _persist(self, run: TraceRun):
        pass

    @cleanup
    def _on_shutdown(self):
        logger.debug("TraceService: flushing %d traces", len(self.traces))
        self.traces.clear()

    def get_traces(self) -> List[Dict[str, Any]]:
        """Return all recorded traces as a list of dictionaries.

        Returns:
            List of dicts, each representing a ``TraceRun``.
        """
        return [asdict(t) for t in self.traces]
