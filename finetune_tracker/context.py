"""Run handle and context-manager wrapper."""

from __future__ import annotations

from datetime import datetime, timezone

from finetune_tracker.client import TrackerClient


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


class Run:
    """A live run handle. Use as a context manager or call finish() manually."""

    def __init__(self, client: TrackerClient, run_id: str) -> None:
        self._client = client
        self.run_id = run_id
        self._finished = False

    def log_hyperparams(self, hyperparams: dict) -> None:
        """Merge hyperparameters into the run record."""
        current = self._client.get_run(self.run_id).get("hyperparameters", {})
        merged = {**current, **hyperparams}
        self._client.update_run(self.run_id, hyperparameters=merged)

    def log_metrics(
        self, metrics: dict[str, float], step: int | None = None, epoch: float | None = None
    ) -> None:
        """Log a dict of metric name -> value at a given step/epoch."""
        batch = [
            {"step": step, "epoch": epoch, "name": name, "value": float(value)}
            for name, value in metrics.items()
        ]
        self._client.log_metrics(self.run_id, batch)

    # Convenience alias used in the prompt example: run.log(...)
    log = log_metrics

    def finish(self, status: str = "completed") -> None:
        if self._finished:
            return
        self._client.update_run(
            self.run_id, status=status, finished_at=_utcnow_iso()
        )
        self._finished = True

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.finish(status="failed" if exc_type is not None else "completed")
        return False  # never suppress exceptions
