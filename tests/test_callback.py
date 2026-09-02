"""Test the HuggingFace callback logs metrics, using a mocked SDK.

Avoids importing transformers: we patch the module flag and base class so the
callback logic can be exercised without the optional dependency installed.
"""

from __future__ import annotations

import types

import pytest


@pytest.fixture()
def callback_cls(monkeypatch):
    import finetune_tracker.callbacks as cb

    monkeypatch.setattr(cb, "_HAS_TRANSFORMERS", True, raising=False)
    return cb.FinetuneTrackerCallback


class _FakeRun:
    def __init__(self):
        self.logged = []
        self.finished = None

    def log_metrics(self, metrics, step=None, epoch=None):
        self.logged.append((metrics, step, epoch))

    def finish(self, status="completed"):
        self.finished = status


def test_callback_logs_losses(callback_cls, monkeypatch):
    import finetune_tracker as ft

    fake_run = _FakeRun()
    monkeypatch.setattr(ft, "get_or_create_experiment", lambda *a, **k: {"id": "x"})
    monkeypatch.setattr(ft, "create_run", lambda *a, **k: fake_run)

    cb = callback_cls(experiment="my-exp")

    args = types.SimpleNamespace(
        learning_rate=1e-4,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        weight_decay=0.01,
        warmup_steps=100,
    )
    state = types.SimpleNamespace(global_step=0, epoch=0.0)

    cb.on_train_begin(args, state, control=None)

    state.global_step = 10
    state.epoch = 1.0
    cb.on_log(args, state, control=None, logs={"loss": 1.5, "eval_loss": 1.2})

    cb.on_train_end(args, state, control=None)

    assert fake_run.logged == [({"train_loss": 1.5, "eval_loss": 1.2}, 10, 1.0)]
    assert fake_run.finished == "completed"
