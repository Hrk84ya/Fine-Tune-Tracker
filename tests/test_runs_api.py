"""Tests for experiment and run endpoints."""

from __future__ import annotations


def test_create_and_get_experiment(client):
    resp = client.post("/api/experiments", json={"name": "llama3-lora"})
    assert resp.status_code == 201
    exp = resp.json()
    assert exp["name"] == "llama3-lora"

    got = client.get(f"/api/experiments/{exp['id']}")
    assert got.status_code == 200
    assert got.json()["run_count"] == 0


def test_duplicate_experiment_returns_409(client):
    client.post("/api/experiments", json={"name": "dup"})
    resp = client.post("/api/experiments", json={"name": "dup"})
    assert resp.status_code == 409


def test_create_run_requires_experiment(client):
    resp = client.post(
        "/api/runs", json={"experiment_id": "does-not-exist", "name": "r"}
    )
    assert resp.status_code == 404


def test_run_lifecycle(client, experiment):
    create = client.post(
        "/api/runs",
        json={
            "experiment_id": experiment["id"],
            "name": "run-a",
            "base_model": "meta-llama/Llama-3-8B",
            "hyperparameters": {"learning_rate": 1e-4, "batch_size": 8},
            "tags": ["baseline"],
        },
    )
    assert create.status_code == 201
    run = create.json()
    assert run["status"] == "running"

    patched = client.patch(
        f"/api/runs/{run['id']}",
        json={"status": "completed", "finished_at": "2026-01-01T00:00:10"},
    )
    assert patched.status_code == 200
    assert patched.json()["status"] == "completed"

    detail = client.get(f"/api/runs/{run['id']}")
    assert detail.status_code == 200
    assert detail.json()["duration_seconds"] is not None


def test_list_runs_filters(client, experiment):
    client.post(
        "/api/runs",
        json={"experiment_id": experiment["id"], "name": "r1", "tags": ["baseline"]},
    )
    client.post(
        "/api/runs",
        json={"experiment_id": experiment["id"], "name": "r2", "tags": ["lora"]},
    )
    all_runs = client.get("/api/runs").json()
    assert len(all_runs) == 2

    by_tag = client.get("/api/runs", params={"tag": "lora"}).json()
    assert len(by_tag) == 1
    assert by_tag[0]["name"] == "r2"

    by_status = client.get("/api/runs", params={"status": "running"}).json()
    assert len(by_status) == 2


def test_delete_run_cascades_metrics(client, experiment):
    run = client.post(
        "/api/runs", json={"experiment_id": experiment["id"], "name": "r"}
    ).json()
    client.post(
        f"/api/runs/{run['id']}/metrics",
        json={"metrics": [{"step": 0, "name": "train_loss", "value": 1.0}]},
    )
    resp = client.delete(f"/api/runs/{run['id']}")
    assert resp.status_code == 204
    assert client.get(f"/api/runs/{run['id']}").status_code == 404


def test_experiment_best_eval_loss(client, experiment):
    run = client.post(
        "/api/runs", json={"experiment_id": experiment["id"], "name": "r"}
    ).json()
    client.post(
        f"/api/runs/{run['id']}/metrics",
        json={
            "metrics": [
                {"step": 0, "name": "eval_loss", "value": 0.9},
                {"step": 1, "name": "eval_loss", "value": 0.4},
            ]
        },
    )
    summary = client.get(f"/api/experiments/{experiment['id']}").json()
    assert summary["run_count"] == 1
    assert summary["best_eval_loss"] == 0.4
