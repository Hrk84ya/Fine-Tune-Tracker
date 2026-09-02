# Fine-tune Tracker

A lightweight, self-hosted tracker for fine-tuning runs — a local-first alternative to the
MLflow Tracking UI, focused on LLM/deep-learning fine-tuning workflows.

Log runs and stream metrics from a training script with a tiny Python SDK, then monitor and
compare them in a Plotly Dash dashboard. Everything runs in a single process on top of
FastAPI + SQLite. No Docker, no JS framework, no external database.

## Features

- **Run logging** via REST API or the `finetune_tracker` Python SDK
- **Metric streaming** — push train/val loss, accuracy, etc. per step or epoch
- **Dashboard** — sortable/filterable runs table, multi-run loss-curve overlay (step or epoch
  x-axis), hyperparameter diff table, per-run detail with editable notes
- **Live auto-refresh** every 10s for running runs
- **Experiments** — group runs, tag them, see run counts and best eval loss
- **Hugging Face** `Trainer` callback for automatic metric logging
- **Export** run metrics as CSV; full REST API for integration

## Requirements

- Python 3.11+
- macOS or Linux

## Install

```bash
python -m venv .venv
source .venv/bin/activate

# core + dev tools (tests)
pip install -e ".[dev]"

# optional: Hugging Face Trainer callback support
pip install -e ".[hf]"
```

Copy the example config if you want to change ports or paths (all values have sane defaults):

```bash
cp .env.example .env
```

## Run

```bash
python backend/main.py
```

This starts the REST API and the dashboard in one process:

- Dashboard: http://127.0.0.1:8000/
- API docs (Swagger): http://127.0.0.1:8000/docs

## Quickstart (SDK)

With the backend running, log a run from any training script:

```python
import finetune_tracker as ft

# Create or get an experiment
exp = ft.get_or_create_experiment("llama3-lora-v2")

# Start a run
with ft.run(experiment=exp, name="lr-1e-4-bs-8") as run:
    run.log_hyperparams({
        "learning_rate": 1e-4,
        "batch_size": 8,
        "epochs": 3,
        "optimizer": "adamw",
        "lora_rank": 16,
    })

    for step, (train_loss, val_loss) in enumerate(training_loop()):
        run.log_metrics({
            "train_loss": train_loss,
            "eval_loss": val_loss,
        }, step=step)

# Run is automatically marked "completed" on context exit,
# or "failed" if an exception is raised.
```

Try the ready-made example:

```bash
python examples/log_dummy_run.py
```

### Hugging Face Trainer

```python
from transformers import Trainer
from finetune_tracker import FinetuneTrackerCallback

trainer = Trainer(
    ...,
    callbacks=[FinetuneTrackerCallback(experiment="my-exp")],
)
trainer.train()
```

See `examples/hf_trainer_example.py` for a runnable end-to-end example.

## API overview

```
POST   /api/experiments                 Create experiment
GET    /api/experiments                 List experiments (with run count + best eval loss)
GET    /api/experiments/{id}            Experiment summary

POST   /api/runs                        Create a run
GET    /api/runs                        List runs (filter by experiment, status, tag)
GET    /api/runs/{id}                    Run detail
PATCH  /api/runs/{id}                    Update status / notes / finished_at / hyperparams
DELETE /api/runs/{id}                    Delete run + its metrics

POST   /api/runs/{id}/metrics            Log a batch of metrics
GET    /api/runs/{id}/metrics            Get all metrics for a run
GET    /api/runs/{id}/metrics/export     Download metrics as CSV

GET    /api/compare?run_ids=a,b,c        Merged metric series for comparison
```

## Tests

```bash
pytest --cov
```

## Project layout

```
finetune_tracker/   Python SDK (client, context manager, HF callback)
backend/            FastAPI app, SQLAlchemy models, routers, scheduler
dashboard/          Plotly Dash app (pages + reusable components)
tests/              pytest suite (API + SDK + callback)
examples/           runnable SDK and HF Trainer examples
```

## Contributing

1. Fork and clone the repo.
2. `pip install -e ".[dev]"`.
3. Make your change with tests. Keep changes surgical and match the existing style.
4. Run `pytest --cov` and make sure it passes.
5. Open a pull request describing what changed and how you verified it.
