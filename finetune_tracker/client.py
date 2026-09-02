"""HTTP client for the Fine-tune Tracker backend."""

from __future__ import annotations

import os

import httpx

DEFAULT_API_URL = os.environ.get("FT_API_URL", "http://127.0.0.1:8000")


class TrackerError(RuntimeError):
    """Raised when the backend returns an error response."""


class TrackerClient:
    """Thin wrapper over the REST API used by the SDK."""

    def __init__(
        self,
        api_url: str | None = None,
        timeout: float = 10.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.api_url = (api_url or DEFAULT_API_URL).rstrip("/")
        # http_client is primarily an injection point for tests (e.g. an
        # httpx.Client backed by an ASGI transport). Defaults to a real client.
        self._http = http_client or httpx.Client(base_url=self.api_url, timeout=timeout)

    # -- experiments ------------------------------------------------------- #
    def create_experiment(self, name: str, description: str | None = None) -> dict:
        resp = self._http.post(
            "/api/experiments", json={"name": name, "description": description}
        )
        return self._json(resp)

    def list_experiments(self) -> list[dict]:
        return self._json(self._http.get("/api/experiments"))

    def get_or_create_experiment(
        self, name: str, description: str | None = None
    ) -> dict:
        """Return the experiment with this name, creating it if needed."""
        for exp in self.list_experiments():
            if exp["name"] == name:
                return exp
        resp = self._http.post(
            "/api/experiments", json={"name": name, "description": description}
        )
        if resp.status_code == httpx.codes.CONFLICT:
            # Race: created concurrently. Re-fetch.
            for exp in self.list_experiments():
                if exp["name"] == name:
                    return exp
        return self._json(resp)

    # -- runs -------------------------------------------------------------- #
    def create_run(
        self,
        experiment_id: str,
        name: str,
        base_model: str | None = None,
        dataset: str | None = None,
        tags: list[str] | None = None,
        hardware: dict | None = None,
        hyperparameters: dict | None = None,
        notes: str | None = None,
    ) -> dict:
        payload = {
            "experiment_id": experiment_id,
            "name": name,
            "base_model": base_model,
            "dataset": dataset,
            "tags": tags or [],
            "hardware": hardware or {},
            "hyperparameters": hyperparameters or {},
            "notes": notes,
        }
        return self._json(self._http.post("/api/runs", json=payload))

    def update_run(self, run_id: str, **fields) -> dict:
        return self._json(self._http.patch(f"/api/runs/{run_id}", json=fields))

    def get_run(self, run_id: str) -> dict:
        return self._json(self._http.get(f"/api/runs/{run_id}"))

    def delete_run(self, run_id: str) -> None:
        resp = self._http.delete(f"/api/runs/{run_id}")
        if resp.status_code >= 400:
            raise TrackerError(f"{resp.status_code}: {resp.text}")

    # -- metrics ----------------------------------------------------------- #
    def log_metrics(self, run_id: str, metrics: list[dict]) -> list[dict]:
        return self._json(
            self._http.post(
                f"/api/runs/{run_id}/metrics", json={"metrics": metrics}
            )
        )

    def get_metrics(self, run_id: str) -> list[dict]:
        return self._json(self._http.get(f"/api/runs/{run_id}/metrics"))

    def compare(self, run_ids: list[str]) -> dict:
        return self._json(
            self._http.get("/api/compare", params={"run_ids": ",".join(run_ids)})
        )

    # -- internals --------------------------------------------------------- #
    @staticmethod
    def _json(resp: httpx.Response):
        if resp.status_code >= 400:
            raise TrackerError(f"{resp.status_code}: {resp.text}")
        return resp.json()

    def close(self) -> None:
        self._http.close()
