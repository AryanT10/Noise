"""LangSmith integration and node-level tracing utilities.

Provides:
  - ``setup_langsmith()`` — configure LangSmith env vars from Settings
  - ``traced_node()``     — decorator that logs + traces each graph node
  - ``get_langsmith_url()`` — return the dashboard URL for the current project
"""

from __future__ import annotations

import functools
import os
import time
from contextvars import ContextVar
from typing import Any, Callable, Awaitable

from app.config import settings
from app.logging import logger
from app.observability.trace_store import trace_store

# ── context var for the current run_id ───────────────────────
_current_run_id: ContextVar[str | None] = ContextVar("_current_run_id", default=None)


def get_current_run_id() -> str | None:
    return _current_run_id.get()


def set_current_run_id(run_id: str | None) -> None:
    _current_run_id.set(run_id)


# ── LangSmith setup ─────────────────────────────────────────


def setup_langsmith() -> bool:
    """Configure LangSmith environment variables.

    Returns True if LangSmith tracing is enabled.
    """
    if not settings.langsmith_api_key:
        logger.info("LangSmith API key not set — tracing disabled")
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key

    if settings.langsmith_project:
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    else:
        os.environ["LANGCHAIN_PROJECT"] = "noise"

    if settings.langsmith_endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint

    logger.info(
        "LangSmith tracing enabled — project=%s",
        os.environ.get("LANGCHAIN_PROJECT"),
    )
    return True


def get_langsmith_url() -> str | None:
    """Return the LangSmith dashboard URL for the project, or None."""
    if not settings.langsmith_api_key:
        return None
    endpoint = settings.langsmith_endpoint or "https://smith.langchain.com"
    project = settings.langsmith_project or "noise"
    return f"{endpoint}/o/default/projects?filter={project}"


# ── Node tracing decorator ──────────────────────────────────


def traced_node(
    fn: Callable[[dict], Awaitable[dict]],
) -> Callable[[dict], Awaitable[dict]]:
    """Wrap an async graph-node function with observability.

    - Logs input keys / output keys & timing
    - Records to the TraceStore
    - Catches and records errors
    """

    @functools.wraps(fn)
    async def wrapper(state: dict) -> dict:
        node_name = fn.__name__
        run_id = get_current_run_id()
        start = time.time()

        # Summarise input for logging (just keys + question prefix)
        input_keys = list(state.keys())
        question_prefix = state.get("question", "")[:60]
        logger.info(
            "▶ Node [%s] start | run=%s | keys=%s | q=%s",
            node_name,
            run_id or "?",
            input_keys,
            question_prefix,
        )

        if run_id:
            trace_store.record_node_start(run_id, node_name, state)

        try:
            result = await fn(state)
            elapsed = round((time.time() - start) * 1000, 2)

            output_keys = list(result.keys()) if isinstance(result, dict) else []
            errors_in_result = result.get("errors", []) if isinstance(result, dict) else []

            logger.info(
                "◀ Node [%s] done  | run=%s | %sms | out_keys=%s | errors=%d",
                node_name,
                run_id or "?",
                elapsed,
                output_keys,
                len(errors_in_result),
            )

            if run_id:
                trace_store.record_node_end(run_id, node_name, result)

            return result

        except Exception as exc:
            elapsed = round((time.time() - start) * 1000, 2)
            logger.error(
                "✖ Node [%s] FAILED | run=%s | %sms | %s: %s",
                node_name,
                run_id or "?",
                elapsed,
                type(exc).__name__,
                exc,
            )
            if run_id:
                trace_store.record_node_end(
                    run_id, node_name, {}, error=f"{type(exc).__name__}: {exc}"
                )
            raise

    return wrapper
