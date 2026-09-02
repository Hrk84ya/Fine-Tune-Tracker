"""Run endpoints (CRUD + summary)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database import get_session
from backend.models import Experiment, Run, RunStatus
from backend.schemas import RunCreate, RunOut, RunSummary, RunUpdate
from backend.summaries import best_eval_loss, duration_seconds

router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.post("", response_model=RunOut, status_code=status.HTTP_201_CREATED)
def create_run(payload: RunCreate, session: Session = Depends(get_session)) -> Run:
    if session.get(Experiment, payload.experiment_id) is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    run = Run(
        experiment_id=payload.experiment_id,
        name=payload.name,
        base_model=payload.base_model,
        dataset=payload.dataset,
        tags=payload.tags,
        hardware=payload.hardware,
        hyperparameters=payload.hyperparameters,
        notes=payload.notes,
        status=RunStatus.running,
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


@router.get("", response_model=list[RunSummary])
def list_runs(
    session: Session = Depends(get_session),
    experiment_id: str | None = Query(default=None),
    run_status: RunStatus | None = Query(default=None, alias="status"),
    tag: str | None = Query(default=None),
) -> list[RunSummary]:
    stmt = select(Run)
    if experiment_id is not None:
        stmt = stmt.where(Run.experiment_id == experiment_id)
    if run_status is not None:
        stmt = stmt.where(Run.status == run_status)
    runs = session.execute(stmt).scalars().all()
    if tag is not None:
        runs = [run for run in runs if tag in (run.tags or [])]
    return [_summarize(session, run) for run in runs]


@router.get("/{run_id}", response_model=RunSummary)
def get_run(run_id: str, session: Session = Depends(get_session)) -> RunSummary:
    run = session.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return _summarize(session, run)


@router.patch("/{run_id}", response_model=RunOut)
def update_run(
    run_id: str, payload: RunUpdate, session: Session = Depends(get_session)
) -> Run:
    run = session.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    data = payload.model_dump(exclude_unset=True)
    for field, value in data.items():
        setattr(run, field, value)
    session.commit()
    session.refresh(run)
    return run


@router.delete("/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_run(run_id: str, session: Session = Depends(get_session)) -> None:
    run = session.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    session.delete(run)
    session.commit()


def _summarize(session: Session, run: Run) -> RunSummary:
    return RunSummary(
        id=run.id,
        experiment_id=run.experiment_id,
        name=run.name,
        base_model=run.base_model,
        dataset=run.dataset,
        status=run.status,
        tags=run.tags or [],
        hardware=run.hardware or {},
        hyperparameters=run.hyperparameters or {},
        notes=run.notes,
        started_at=run.started_at,
        finished_at=run.finished_at,
        duration_seconds=duration_seconds(run),
        best_eval_loss=best_eval_loss(session, run.id),
    )
