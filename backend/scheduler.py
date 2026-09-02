"""Background cleanup of stale 'running' runs via APScheduler."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import func, select

from backend.config import settings
from backend.database import SessionLocal
from backend.models import Metric, Run, RunStatus


def cleanup_stale_runs() -> int:
    """Mark running runs with no recent activity as failed. Returns count updated."""
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
        minutes=settings.stale_minutes
    )
    updated = 0
    with SessionLocal() as session:
        running = session.execute(
            select(Run).where(Run.status == RunStatus.running)
        ).scalars().all()
        for run in running:
            last_metric = session.execute(
                select(func.max(Metric.logged_at)).where(Metric.run_id == run.id)
            ).scalar_one_or_none()
            last_activity = last_metric or run.started_at
            if last_activity < cutoff:
                run.status = RunStatus.failed
                run.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
                updated += 1
        session.commit()
    return updated


def start_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(cleanup_stale_runs, "interval", minutes=1, id="stale_cleanup")
    scheduler.start()
    return scheduler
