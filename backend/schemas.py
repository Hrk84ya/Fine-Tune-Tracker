"""Pydantic request/response schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from backend.models import RunStatus


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
class ExperimentCreate(BaseModel):
    name: str
    description: str | None = None


class ExperimentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: str | None
    created_at: datetime


class ExperimentSummary(ExperimentOut):
    run_count: int
    best_eval_loss: float | None = None


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #
class RunCreate(BaseModel):
    experiment_id: str
    name: str
    base_model: str | None = None
    dataset: str | None = None
    tags: list[str] = Field(default_factory=list)
    hardware: dict = Field(default_factory=dict)
    hyperparameters: dict = Field(default_factory=dict)
    notes: str | None = None


class RunUpdate(BaseModel):
    status: RunStatus | None = None
    notes: str | None = None
    finished_at: datetime | None = None
    hyperparameters: dict | None = None


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    experiment_id: str
    name: str
    base_model: str | None
    dataset: str | None
    status: RunStatus
    tags: list[str]
    hardware: dict
    hyperparameters: dict
    notes: str | None
    started_at: datetime
    finished_at: datetime | None


class RunSummary(RunOut):
    duration_seconds: float | None = None
    best_eval_loss: float | None = None


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
class MetricIn(BaseModel):
    step: int | None = None
    epoch: float | None = None
    name: str
    value: float


class MetricBatch(BaseModel):
    metrics: list[MetricIn]


class MetricOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: str
    step: int | None
    epoch: float | None
    name: str
    value: float
    logged_at: datetime


class MetricPoint(BaseModel):
    step: int | None
    epoch: float | None
    value: float


class CompareOut(BaseModel):
    """Merged metric series keyed by run id then metric name."""

    runs: dict[str, dict[str, list[MetricPoint]]]
