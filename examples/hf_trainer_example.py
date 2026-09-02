"""HuggingFace Trainer + FinetuneTrackerCallback example.

Requires the optional 'hf' extra and the backend running:

    pip install -e ".[hf]"
    python backend/main.py          # in another terminal
    python examples/hf_trainer_example.py

This is a minimal, self-contained example using a tiny model and dummy data so
it runs quickly on CPU. Swap in your real model/dataset for actual training.
"""

from __future__ import annotations


def main() -> None:
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    from finetune_tracker import FinetuneTrackerCallback

    model_name = "prajjwal1/bert-tiny"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    texts = ["great movie", "terrible film", "loved it", "awful and boring"] * 8
    labels = [1, 0, 1, 0] * 8
    ds = Dataset.from_dict({"text": texts, "label": labels})

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=16)

    ds = ds.map(tokenize, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"eval_accuracy": float((preds == labels).mean())}

    args = TrainingArguments(
        output_dir="./hf_out",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        learning_rate=5e-4,
        logging_steps=1,
        eval_strategy="epoch",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        eval_dataset=ds,
        compute_metrics=compute_metrics,
        callbacks=[FinetuneTrackerCallback(experiment="bert-tiny-demo")],
    )
    trainer.train()
    print("Training complete. Open http://127.0.0.1:8000 to view the run.")


if __name__ == "__main__":
    main()
