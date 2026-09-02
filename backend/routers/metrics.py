"""Metric ingestion, retrieval, export, and comparison endpoints."""

from __future__ import annotations

import csv
import io

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database import get_session
from backend.models import Metric, Run
from backend.schemas import (
    CompareOut,
    MetricBatch,
    MetricOut,
    MetricPoint,
)

router = APIRouter(prefix="/api", tags=["metrics"])


@router.post(
    "/runs/{run_id}/metrics",
    response_model=list[MetricOut],
    status_code=status.HTTP_201_CREATED,
)
def log_metrics(
    run_id: str, payload: MetricBatch, session: Session = Depends(get_session)
) -> list[Metric]:
    if session.get(Run, run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")
    created = [
        Metric(
            run_id=run_id,
            step=m.step,
            epoch=m.epoch,
            name=m.name,
            value=m.value,
        )
        for m in payload.metrics
    ]
    session.add_all(created)
    session.commit()
    for metric in created:
        session.refresh(metric)
    return created


@router.get("/runs/{run_id}/metrics", response_model=list[MetricOut])
def get_metrics(
    run_id: str,
    session: Session = Depends(get_session),
    name: str | None = Query(default=None),
) -> list[Metric]:
    if session.get(Run, run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")
    stmt = select(Metric).where(Metric.run_id == run_id)
    if name is not None:
        stmt = stmt.where(Metric.name == name)
    stmt = stmt.order_by(Metric.id)
    return session.execute(stmt).scalars().all()


@router.get("/runs/{run_id}/metrics/export")
def export_metrics(run_id: str, session: Session = Depends(get_session)) -> StreamingResponse:
    if session.get(Run, run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")
    rows = session.execute(
        select(Metric).where(Metric.run_id == run_id).order_by(Metric.id)
    ).scalars().all()

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["step", "epoch", "name", "value", "logged_at"])
    for m in rows:
        writer.writerow(
            [m.step, m.epoch, m.name, m.value, m.logged_at.isoformat()]
        )
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="run_{run_id}_metrics.csv"'
        },
    )


@router.get("/compare", response_model=CompareOut)
def compare_runs(
    run_ids: str = Query(..., description="Comma-separated run ids"),
    session: Session = Depends(get_session),
) -> CompareOut:
    ids = [rid.strip() for rid in run_ids.split(",") if rid.strip()]
    merged: dict[str, dict[str, list[MetricPoint]]] = {}
    for run_id in ids:
        metrics = session.execute(
            select(Metric).where(Metric.run_id == run_id).order_by(Metric.id)
        ).scalars().all()
        by_name: dict[str, list[MetricPoint]] = {}
        for m in metrics:
            by_name.setdefault(m.name, []).append(
                MetricPoint(step=m.step, epoch=m.epoch, value=m.value)
            )
        merged[run_id] = by_name
    return CompareOut(runs=merged)
