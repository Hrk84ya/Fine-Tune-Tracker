"""Thin HTTP helper the dashboard uses to reach the REST API.

The dashboard is mounted on the same process as the API, so it talks to the
local server over HTTP. This keeps the API as the single source of truth.
"""

from __future__ import annotations

import httpx

from backend.config import settings

_BASE = settings.api_url.rstrip("/")


def _get(path: str, **params) -> object:
    with httpx.Client(base_url=_BASE, timeout=10.0) as client:
        resp = client.get(path, params={k: v for k, v in params.items() if v is not None})
        resp.raise_for_status()
        return resp.json()


def list_experiments() -> list[dict]:
    return _get("/api/experiments")  # type: ignore[return-value]


def list_runs(experiment_id: str | None = None) -> list[dict]:
    return _get("/api/runs", experiment_id=experiment_id)  # type: ignore[return-value]


def get_run(run_id: str) -> dict:
    return _get(f"/api/runs/{run_id}")  # type: ignore[return-value]


def get_metrics(run_id: str) -> list[dict]:
    return _get(f"/api/runs/{run_id}/metrics")  # type: ignore[return-value]


def compare(run_ids: list[str]) -> dict:
    return _get("/api/compare", run_ids=",".join(run_ids))  # type: ignore[return-value]


def update_run(run_id: str, **fields) -> dict:
    with httpx.Client(base_url=_BASE, timeout=10.0) as client:
        resp = client.patch(f"/api/runs/{run_id}", json=fields)
        resp.raise_for_status()
        return resp.json()
