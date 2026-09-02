"""Tests for the finetune_tracker Python SDK against an in-process API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from finetune_tracker.context import Run
from finetune_tracker.client import TrackerClient


@pytest.fixture()
def sdk_client(app) -> TrackerClient:
    """A TrackerClient backed by the in-process FastAPI app (no network).

    FastAPI's TestClient is a sync httpx.Client that correctly drives the ASGI
    app, so we inject it directly as the SDK's HTTP client.
    """
    return TrackerClient(http_client=TestClient(app))


def test_full_run_lifecycle_via_sdk(sdk_client):
    exp = sdk_client.get_or_create_experiment("sdk-exp")
    created = sdk_client.create_run(experiment_id=exp["id"], name="sdk-run")
    run = Run(sdk_client, created["id"])

    run.log_hyperparams({"learning_rate": 1e-4, "batch_size": 8})
    for step in range(3):
        run.log_metrics(
            {"train_loss": 2.0 - step * 0.5, "eval_loss": 1.9 - step * 0.4},
            step=step,
            epoch=float(step),
        )
    run.finish()

    detail = sdk_client.get_run(created["id"])
    assert detail["status"] == "completed"
    assert detail["finished_at"] is not None
    assert detail["hyperparameters"]["batch_size"] == 8

    metrics = sdk_client.get_metrics(created["id"])
    assert len([m for m in metrics if m["name"] == "train_loss"]) == 3


def test_get_or_create_is_idempotent(sdk_client):
    first = sdk_client.get_or_create_experiment("same-name")
    second = sdk_client.get_or_create_experiment("same-name")
    assert first["id"] == second["id"]


def test_context_manager_marks_completed(sdk_client):
    exp = sdk_client.get_or_create_experiment("ctx-exp")
    created = sdk_client.create_run(experiment_id=exp["id"], name="ctx-run")
    with Run(sdk_client, created["id"]) as run:
        run.log_metrics({"train_loss": 1.0}, step=0)
    assert sdk_client.get_run(created["id"])["status"] == "completed"


def test_context_manager_marks_failed_on_exception(sdk_client):
    exp = sdk_client.get_or_create_experiment("fail-exp")
    created = sdk_client.create_run(experiment_id=exp["id"], name="fail-run")
    with pytest.raises(ValueError):
        with Run(sdk_client, created["id"]) as run:
            run.log_metrics({"train_loss": 1.0}, step=0)
            raise ValueError("boom")
    assert sdk_client.get_run(created["id"])["status"] == "failed"


def test_compare_via_sdk(sdk_client):
    exp = sdk_client.get_or_create_experiment("cmp-exp")
    ids = []
    for i in range(2):
        created = sdk_client.create_run(experiment_id=exp["id"], name=f"r{i}")
        ids.append(created["id"])
        sdk_client.log_metrics(
            created["id"], [{"step": 0, "name": "train_loss", "value": float(i)}]
        )
    data = sdk_client.compare(ids)["runs"]
    assert set(data.keys()) == set(ids)
