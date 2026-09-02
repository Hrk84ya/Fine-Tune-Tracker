"""Experiment endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database import get_session
from backend.models import Experiment, Run
from backend.schemas import ExperimentCreate, ExperimentOut, ExperimentSummary
from backend.summaries import best_eval_loss

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


@router.post("", response_model=ExperimentOut, status_code=status.HTTP_201_CREATED)
def create_experiment(
    payload: ExperimentCreate, session: Session = Depends(get_session)
) -> Experiment:
    existing = session.execute(
        select(Experiment).where(Experiment.name == payload.name)
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Experiment '{payload.name}' already exists",
        )
    experiment = Experiment(name=payload.name, description=payload.description)
    session.add(experiment)
    session.commit()
    session.refresh(experiment)
    return experiment


@router.get("", response_model=list[ExperimentSummary])
def list_experiments(session: Session = Depends(get_session)) -> list[ExperimentSummary]:
    experiments = session.execute(select(Experiment)).scalars().all()
    return [_summarize(session, exp) for exp in experiments]


@router.get("/{experiment_id}", response_model=ExperimentSummary)
def get_experiment(
    experiment_id: str, session: Session = Depends(get_session)
) -> ExperimentSummary:
    experiment = session.get(Experiment, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return _summarize(session, experiment)


def _summarize(session: Session, experiment: Experiment) -> ExperimentSummary:
    run_ids = session.execute(
        select(Run.id).where(Run.experiment_id == experiment.id)
    ).scalars().all()
    losses = [
        loss
        for run_id in run_ids
        if (loss := best_eval_loss(session, run_id)) is not None
    ]
    return ExperimentSummary(
        id=experiment.id,
        name=experiment.name,
        description=experiment.description,
        created_at=experiment.created_at,
        run_count=len(run_ids),
        best_eval_loss=min(losses) if losses else None,
    )
