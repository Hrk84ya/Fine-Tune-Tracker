"""finetune_tracker — minimal Python SDK for the Fine-tune Tracker backend.

Quickstart::

    import finetune_tracker as ft

    exp = ft.get_or_create_experiment("llama3-lora-v2")
    with ft.run(experiment=exp, name="lr-1e-4-bs-8") as run:
        run.log_hyperparams({"learning_rate": 1e-4, "batch_size": 8})
        for step, (train_loss, val_loss) in enumerate(training_loop()):
            run.log_metrics({"train_loss": train_loss, "eval_loss": val_loss}, step=step)
"""

from __future__ import annotations

from finetune_tracker.client import TrackerClient, TrackerError
from finetune_tracker.context import Run

__all__ = [
    "TrackerClient",
    "TrackerError",
    "Run",
    "get_or_create_experiment",
    "create_experiment",
    "create_run",
    "run",
    "FinetuneTrackerCallback",
]


def _client(api_url: str | None) -> TrackerClient:
    return TrackerClient(api_url=api_url)


def get_or_create_experiment(
    name: str, description: str | None = None, api_url: str | None = None
) -> dict:
    """Return an experiment by name, creating it if it doesn't exist."""
    return _client(api_url).get_or_create_experiment(name, description)


def create_experiment(
    name: str, description: str | None = None, api_url: str | None = None
) -> dict:
    """Create a new experiment (raises TrackerError on duplicate name)."""
    return _client(api_url).create_experiment(name, description)


def _experiment_id(experiment: str | dict) -> str:
    return experiment["id"] if isinstance(experiment, dict) else experiment


def create_run(
    experiment: str | dict,
    name: str,
    base_model: str | None = None,
    dataset: str | None = None,
    tags: list[str] | None = None,
    hardware: dict | None = None,
    hyperparameters: dict | None = None,
    notes: str | None = None,
    api_url: str | None = None,
) -> Run:
    """Create a run and return a :class:`Run` handle.

    ``experiment`` may be an experiment id string or the dict returned by
    :func:`get_or_create_experiment`.
    """
    client = _client(api_url)
    created = client.create_run(
        experiment_id=_experiment_id(experiment),
        name=name,
        base_model=base_model,
        dataset=dataset,
        tags=tags,
        hardware=hardware,
        hyperparameters=hyperparameters,
        notes=notes,
    )
    return Run(client, created["id"])


def run(
    experiment: str | dict,
    name: str,
    api_url: str | None = None,
    **kwargs,
) -> Run:
    """Create a run for use as a context manager.

    ``with ft.run(experiment=exp, name="...") as r: ...``
    """
    return create_run(experiment=experiment, name=name, api_url=api_url, **kwargs)


def __getattr__(attr: str):
    # Lazy import so the HF callback (and transformers) is only loaded on demand.
    if attr == "FinetuneTrackerCallback":
        from finetune_tracker.callbacks import FinetuneTrackerCallback

        return FinetuneTrackerCallback
    raise AttributeError(f"module 'finetune_tracker' has no attribute '{attr}'")
