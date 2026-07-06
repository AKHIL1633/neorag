"""
Observability — structured JSON tracing for LLM calls, correlated with a
per-request ID set by the middleware in app.main.
"""

import json
import time
import uuid
from contextvars import ContextVar

from loguru import logger

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


class LLMCallTrace:
    """Context manager that logs one structured JSON line per LLM call:
    trace_id, request_id, operation, model, question (truncated), latency_ms,
    success, and anything set via `.set(**kv)` (e.g. tokens_out, judge_score).
    """

    def __init__(self, operation: str, model: str, question: str):
        self.trace = {
            "trace_id": uuid.uuid4().hex[:16],
            "request_id": request_id_ctx.get(),
            "operation": operation,
            "model": model,
            "question": question[:200],
            "start_ts": time.time(),
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.trace["latency_ms"] = round((time.time() - self.trace["start_ts"]) * 1000, 2)
        self.trace["success"] = exc_type is None
        if exc_type:
            self.trace["error"] = str(exc)
        logger.info(json.dumps(self.trace))

    def set(self, **kv):
        self.trace.update(kv)
