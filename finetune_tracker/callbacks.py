"""Hugging Face Trainer callback that logs metrics to Fine-tune Tracker.

The ``transformers`` package is an optional dependency (extra: ``hf``). This
module imports it lazily so the rest of the SDK works without it installed.
"""

from __future__ import annotations

from typing import Any

try:
    from transformers import TrainerCallback

    _HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover - exercised only without transformers
    TrainerCallback = object  # type: ignore[assignment,misc]
    _HAS_TRANSFORMERS = False

import finetune_tracker as ft


class FinetuneTrackerCallback(TrainerCallback):
    """Logs train_loss and eval_loss to the tracker during ``Trainer`` runs.

    Usage::

        Trainer(callbacks=[FinetuneTrackerCallback(experiment="my-exp")])
    """

    def __init__(
        self,
        experiment: str,
        run_name: str | None = None,
        api_url: str | None = None,
    ) -> None:
        if not _HAS_TRANSFORMERS:
            raise ImportError(
                "transformers is required for FinetuneTrackerCallback. "
                'Install it with: pip install "finetune-tracker[hf]"'
            )
        self._experiment_name = experiment
        self._run_name = run_name
        self._api_url = api_url
        self._run: ft.Run | None = None

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs) -> None:
        exp = ft.get_or_create_experiment(self._experiment_name, api_url=self._api_url)
        name = self._run_name or f"run-{state.global_step}"
        hyperparams = {
            "learning_rate": getattr(args, "learning_rate", None),
            "epochs": getattr(args, "num_train_epochs", None),
            "batch_size": getattr(args, "per_device_train_batch_size", None),
            "weight_decay": getattr(args, "weight_decay", None),
            "warmup_steps": getattr(args, "warmup_steps", None),
        }
        self._run = ft.create_run(
            experiment=exp,
            name=name,
            hyperparameters={k: v for k, v in hyperparams.items() if v is not None},
            api_url=self._api_url,
        )

    def on_log(self, args: Any, state: Any, control: Any, logs=None, **kwargs) -> None:
        if self._run is None or not logs:
            return
        tracked = {}
        if "loss" in logs:
            tracked["train_loss"] = logs["loss"]
        if "eval_loss" in logs:
            tracked["eval_loss"] = logs["eval_loss"]
        if "eval_accuracy" in logs:
            tracked["eval_accuracy"] = logs["eval_accuracy"]
        if tracked:
            self._run.log_metrics(
                tracked, step=state.global_step, epoch=state.epoch
            )

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs) -> None:
        if self._run is not None:
            self._run.finish(status="completed")
