"""Trace store — captures intermediate graph states for debugging.

Each graph invocation gets a unique run_id.  The store keeps a bounded
number of recent runs in memory so you can inspect what happened.
"""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

MAX_RUNS = 50  # keep the last N runs in memory


@dataclass
class NodeTrace:
    """Snapshot of one node execution."""

    node: str
    started_at: float
    finished_at: float | None = None
    duration_ms: float | None = None
    input_state: dict[str, Any] = field(default_factory=dict)
    output_patch: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class RunTrace:
    """Full trace for a single graph invocation."""

    run_id: str
    question: str
    started_at: float
    finished_at: float | None = None
    duration_ms: float | None = None
    nodes: list[NodeTrace] = field(default_factory=list)
    final_state: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    success: bool = True


class TraceStore:
    """In-memory bounded store of recent run traces."""

    def __init__(self, max_runs: int = MAX_RUNS) -> None:
        self._max_runs = max_runs
        self._runs: OrderedDict[str, RunTrace] = OrderedDict()

    # ── write API ────────────────────────────────────────────

    def start_run(self, question: str) -> str:
        run_id = uuid.uuid4().hex[:12]
        trace = RunTrace(
            run_id=run_id,
            question=question,
            started_at=time.time(),
        )
        self._runs[run_id] = trace
        self._evict()
        return run_id

    def record_node_start(self, run_id: str, node: str, input_state: dict) -> None:
        trace = self._runs.get(run_id)
        if not trace:
            return
        trace.nodes.append(
            NodeTrace(
                node=node,
                started_at=time.time(),
                input_state=_safe_serialize(input_state),
            )
        )

    def record_node_end(
        self, run_id: str, node: str, output_patch: dict, error: str | None = None
    ) -> None:
        trace = self._runs.get(run_id)
        if not trace:
            return
        # Find the most recent in-progress entry for this node
        for nt in reversed(trace.nodes):
            if nt.node == node and nt.finished_at is None:
                nt.finished_at = time.time()
                nt.duration_ms = round((nt.finished_at - nt.started_at) * 1000, 2)
                nt.output_patch = _safe_serialize(output_patch)
                nt.error = error
                if error:
                    trace.errors.append(f"{node}: {error}")
                    trace.success = False
                break

    def finish_run(self, run_id: str, final_state: dict) -> None:
        trace = self._runs.get(run_id)
        if not trace:
            return
        trace.finished_at = time.time()
        trace.duration_ms = round((trace.finished_at - trace.started_at) * 1000, 2)
        trace.final_state = _safe_serialize(final_state)

    # ── read API ─────────────────────────────────────────────

    def get_run(self, run_id: str) -> RunTrace | None:
        return self._runs.get(run_id)

    def list_runs(self, limit: int = 20) -> list[dict]:
        """Return summaries of recent runs, newest first."""
        items = list(self._runs.values())[-limit:]
        items.reverse()
        return [
            {
                "run_id": r.run_id,
                "question": r.question[:120],
                "started_at": r.started_at,
                "duration_ms": r.duration_ms,
                "node_count": len(r.nodes),
                "error_count": len(r.errors),
                "success": r.success,
            }
            for r in items
        ]

    def get_run_detail(self, run_id: str) -> dict | None:
        trace = self._runs.get(run_id)
        if not trace:
            return None
        return {
            "run_id": trace.run_id,
            "question": trace.question,
            "started_at": trace.started_at,
            "finished_at": trace.finished_at,
            "duration_ms": trace.duration_ms,
            "success": trace.success,
            "errors": trace.errors,
            "nodes": [
                {
                    "node": nt.node,
                    "started_at": nt.started_at,
                    "finished_at": nt.finished_at,
                    "duration_ms": nt.duration_ms,
                    "input_keys": list(nt.input_state.keys()),
                    "output_keys": list(nt.output_patch.keys()),
                    "error": nt.error,
                }
                for nt in trace.nodes
            ],
        }

    def get_node_detail(self, run_id: str, node_name: str) -> list[dict] | None:
        """Return full input/output for a specific node in a run."""
        trace = self._runs.get(run_id)
        if not trace:
            return None
        return [
            {
                "node": nt.node,
                "duration_ms": nt.duration_ms,
                "input_state": nt.input_state,
                "output_patch": nt.output_patch,
                "error": nt.error,
            }
            for nt in trace.nodes
            if nt.node == node_name
        ]

    # ── internal ─────────────────────────────────────────────

    def _evict(self) -> None:
        while len(self._runs) > self._max_runs:
            self._runs.popitem(last=False)


# ── helpers ──────────────────────────────────────────────────


def _safe_serialize(obj: Any, max_str_len: int = 2000) -> Any:
    """Make an object JSON-safe, truncating long strings."""
    if isinstance(obj, dict):
        return {k: _safe_serialize(v, max_str_len) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v, max_str_len) for v in obj]
    if isinstance(obj, str) and len(obj) > max_str_len:
        return obj[:max_str_len] + f"…[truncated, {len(obj)} chars total]"
    if isinstance(obj, (int, float, bool, type(None))):
        return obj
    # Pydantic models, dataclasses, etc.
    if hasattr(obj, "model_dump"):
        return _safe_serialize(obj.model_dump(), max_str_len)
    if hasattr(obj, "__dict__"):
        return _safe_serialize(obj.__dict__, max_str_len)
    return str(obj)[:max_str_len]


# Module-level singleton
trace_store = TraceStore()
