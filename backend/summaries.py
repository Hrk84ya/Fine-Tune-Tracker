"""Helpers for deriving run/experiment summary values."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.models import Metric, Run

# Metric names treated as "eval loss", lower is better. First match wins.
EVAL_LOSS_NAMES = ("eval_loss", "val_loss")


def best_eval_loss(session: Session, run_id: str) -> float | None:
    """Return the minimum eval/val loss recorded for a run, if any."""
    for metric_name in EVAL_LOSS_NAMES:
        result = session.execute(
            select(func.min(Metric.value)).where(
                Metric.run_id == run_id, Metric.name == metric_name
            )
        ).scalar_one_or_none()
        if result is not None:
            return result
    return None


def duration_seconds(run: Run) -> float | None:
    """Elapsed seconds between start and finish, or None if still running."""
    if run.finished_at is None:
        return None
    return (run.finished_at - run.started_at).total_seconds()
