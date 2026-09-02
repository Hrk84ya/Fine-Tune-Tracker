"""Log a dummy fine-tuning run using only the Python SDK.

Requires the backend running (python backend/main.py). Then:

    python examples/log_dummy_run.py
"""

from __future__ import annotations

import math
import random
import time

import finetune_tracker as ft


def fake_training_loop(steps: int = 30):
    """Yield decreasing (train_loss, val_loss) with some noise."""
    for step in range(steps):
        train_loss = 2.5 * math.exp(-step / 12) + random.uniform(0, 0.05)
        val_loss = 2.6 * math.exp(-step / 11) + random.uniform(0, 0.08)
        yield train_loss, val_loss


def main() -> None:
    exp = ft.get_or_create_experiment("llama3-lora-v2")

    with ft.run(
        experiment=exp,
        name="lr-1e-4-bs-8",
        base_model="meta-llama/Llama-3-8B",
        dataset="alpaca-cleaned",
        tags=["baseline", "with-lora"],
        hardware={"gpu_type": "A100", "gpu_count": 1, "cpu_cores": 16, "ram_gb": 128},
    ) as run:
        run.log_hyperparams(
            {
                "learning_rate": 1e-4,
                "batch_size": 8,
                "epochs": 3,
                "optimizer": "adamw",
                "lora_rank": 16,
            }
        )
        for step, (train_loss, val_loss) in enumerate(fake_training_loop()):
            run.log_metrics(
                {"train_loss": train_loss, "eval_loss": val_loss},
                step=step,
                epoch=step / 10,
            )
            time.sleep(0.05)

    print("Logged run to experiment 'llama3-lora-v2'. Open http://127.0.0.1:8000 to view.")


if __name__ == "__main__":
    main()
