"""Tests for Phase 8 — observability and tracing."""

import pytest

from app.observability.trace_store import TraceStore, _safe_serialize
from app.observability.tracing import traced_node, set_current_run_id


# ── TraceStore unit tests ────────────────────────────────────


class TestTraceStore:
    def test_start_and_finish_run(self):
        store = TraceStore(max_runs=5)
        run_id = store.start_run("What is X?")
        assert run_id
        assert store.get_run(run_id) is not None
        assert store.get_run(run_id).question == "What is X?"

        store.finish_run(run_id, {"answer": "42"})
        trace = store.get_run(run_id)
        assert trace.finished_at is not None
        assert trace.duration_ms is not None
        assert trace.final_state["answer"] == "42"

    def test_node_recording(self):
        store = TraceStore()
        run_id = store.start_run("test")

        store.record_node_start(run_id, "analyze", {"question": "test"})
        store.record_node_end(run_id, "analyze", {"queries": ["q1"]})

        trace = store.get_run(run_id)
        assert len(trace.nodes) == 1
        assert trace.nodes[0].node == "analyze"
        assert trace.nodes[0].duration_ms is not None
        assert trace.nodes[0].output_patch == {"queries": ["q1"]}

    def test_node_error_recording(self):
        store = TraceStore()
        run_id = store.start_run("test")

        store.record_node_start(run_id, "bad_node", {})
        store.record_node_end(run_id, "bad_node", {}, error="boom")

        trace = store.get_run(run_id)
        assert trace.nodes[0].error == "boom"
        assert not trace.success
        assert "bad_node: boom" in trace.errors

    def test_list_runs(self):
        store = TraceStore(max_runs=10)
        ids = []
        for i in range(5):
            ids.append(store.start_run(f"q{i}"))
            store.finish_run(ids[-1], {})

        listing = store.list_runs(limit=3)
        assert len(listing) == 3
        # newest first
        assert listing[0]["run_id"] == ids[4]

    def test_eviction(self):
        store = TraceStore(max_runs=3)
        ids = []
        for i in range(5):
            ids.append(store.start_run(f"q{i}"))

        # Only the last 3 should remain
        assert store.get_run(ids[0]) is None
        assert store.get_run(ids[1]) is None
        assert store.get_run(ids[2]) is not None
        assert store.get_run(ids[4]) is not None

    def test_get_run_detail(self):
        store = TraceStore()
        run_id = store.start_run("detail test")
        store.record_node_start(run_id, "n1", {"k": "v"})
        store.record_node_end(run_id, "n1", {"out": "val"})
        store.finish_run(run_id, {})

        detail = store.get_run_detail(run_id)
        assert detail["run_id"] == run_id
        assert detail["success"] is True
        assert len(detail["nodes"]) == 1
        assert detail["nodes"][0]["input_keys"] == ["k"]
        assert detail["nodes"][0]["output_keys"] == ["out"]

    def test_get_node_detail(self):
        store = TraceStore()
        run_id = store.start_run("node detail test")
        store.record_node_start(run_id, "my_node", {"a": 1})
        store.record_node_end(run_id, "my_node", {"b": 2})

        entries = store.get_node_detail(run_id, "my_node")
        assert len(entries) == 1
        assert entries[0]["input_state"] == {"a": 1}
        assert entries[0]["output_patch"] == {"b": 2}

    def test_get_node_detail_not_found(self):
        store = TraceStore()
        assert store.get_node_detail("nope", "x") is None

        run_id = store.start_run("test")
        result = store.get_node_detail(run_id, "nonexistent")
        assert result == []


# ── _safe_serialize tests ────────────────────────────────────


class TestSafeSerialize:
    def test_truncates_long_strings(self):
        result = _safe_serialize("x" * 5000, max_str_len=100)
        assert len(result) < 200
        assert "truncated" in result

    def test_handles_nested_dicts(self):
        data = {"a": {"b": [1, 2, "hello"]}}
        assert _safe_serialize(data) == {"a": {"b": [1, 2, "hello"]}}

    def test_handles_none(self):
        assert _safe_serialize(None) is None

    def test_pydantic_model(self):
        from app.models.schemas import Source

        src = Source(number=1, title="T", url="http://x.com")
        result = _safe_serialize(src)
        assert result["number"] == 1
        assert result["title"] == "T"


# ── traced_node decorator test ───────────────────────────────


class TestTracedNode:
    @pytest.mark.asyncio
    async def test_wraps_function(self):
        from app.observability.trace_store import trace_store

        @traced_node
        async def my_node(state: dict) -> dict:
            return {"result": state["question"] + "!"}

        run_id = trace_store.start_run("deco test")
        set_current_run_id(run_id)
        try:
            out = await my_node({"question": "hi"})
            assert out == {"result": "hi!"}

            trace = trace_store.get_run(run_id)
            assert len(trace.nodes) == 1
            assert trace.nodes[0].node == "my_node"
            assert trace.nodes[0].error is None
        finally:
            set_current_run_id(None)

    @pytest.mark.asyncio
    async def test_records_errors(self):
        from app.observability.trace_store import trace_store

        @traced_node
        async def failing_node(state: dict) -> dict:
            raise ValueError("test error")

        run_id = trace_store.start_run("error test")
        set_current_run_id(run_id)
        try:
            with pytest.raises(ValueError, match="test error"):
                await failing_node({"question": "hi"})

            trace = trace_store.get_run(run_id)
            assert len(trace.nodes) == 1
            assert "test error" in trace.nodes[0].error
            assert not trace.success
        finally:
            set_current_run_id(None)
