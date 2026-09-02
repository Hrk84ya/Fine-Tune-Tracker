"""Tests for metric ingestion, retrieval, export, and comparison."""

from __future__ import annotations


def _make_run(client, experiment, name="run"):
    return client.post(
        "/api/runs", json={"experiment_id": experiment["id"], "name": name}
    ).json()


def test_log_and_get_metrics(client, experiment):
    run = _make_run(client, experiment)
    resp = client.post(
        f"/api/runs/{run['id']}/metrics",
        json={
            "metrics": [
                {"step": 0, "epoch": 0.0, "name": "train_loss", "value": 2.5},
                {"step": 1, "epoch": 0.5, "name": "train_loss", "value": 1.8},
            ]
        },
    )
    assert resp.status_code == 201
    assert len(resp.json()) == 2

    metrics = client.get(f"/api/runs/{run['id']}/metrics").json()
    assert [m["value"] for m in metrics] == [2.5, 1.8]


def test_log_metrics_missing_run_404(client):
    resp = client.post(
        "/api/runs/nope/metrics",
        json={"metrics": [{"step": 0, "name": "train_loss", "value": 1.0}]},
    )
    assert resp.status_code == 404


def test_export_metrics_csv(client, experiment):
    run = _make_run(client, experiment)
    client.post(
        f"/api/runs/{run['id']}/metrics",
        json={"metrics": [{"step": 0, "name": "train_loss", "value": 1.0}]},
    )
    resp = client.get(f"/api/runs/{run['id']}/metrics/export")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    body = resp.text
    assert "step,epoch,name,value,logged_at" in body
    assert "train_loss" in body


def test_compare_three_runs(client, experiment):
    run_ids = []
    for i in range(3):
        run = _make_run(client, experiment, name=f"run-{i}")
        run_ids.append(run["id"])
        client.post(
            f"/api/runs/{run['id']}/metrics",
            json={
                "metrics": [
                    {"step": 0, "name": "train_loss", "value": 2.0 + i},
                    {"step": 1, "name": "train_loss", "value": 1.0 + i},
                ]
            },
        )
    resp = client.get("/api/compare", params={"run_ids": ",".join(run_ids)})
    assert resp.status_code == 200
    data = resp.json()["runs"]
    assert set(data.keys()) == set(run_ids)
    for rid in run_ids:
        assert len(data[rid]["train_loss"]) == 2
