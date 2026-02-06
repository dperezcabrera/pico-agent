import time
import uuid
from typing import Dict, Any, Optional, List
from contextvars import ContextVar
from dataclasses import dataclass, field, asdict
from pico_ioc import component, cleanup
from .logging import get_logger

logger = get_logger(__name__)

run_context = ContextVar("run_context", default=None)

@dataclass
class TraceRun:
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
    def __init__(self):
        self.traces: List[TraceRun] = []

    def start_run(self, name: str, run_type: str, inputs: Dict[str, Any], extra: Dict[str, Any] = None) -> str:
        parent_id = run_context.get()
        run_id = str(uuid.uuid4())
        
        run = TraceRun(
            id=run_id,
            name=name,
            run_type=run_type,
            inputs=inputs,
            parent_id=parent_id,
            extra=extra or {}
        )
        
        self.traces.append(run)
        run_context.set(run_id)
        return run_id

    def end_run(self, run_id: str, outputs: Any = None, error: Exception = None):
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
        return [asdict(t) for t in self.traces]
