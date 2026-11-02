"""Opt-in real Hugging Face fine-tuning. Downloads pretrained weights; CPU runs slowly.

Usage: uv run --extra transformers python -m taxonomyiq.fine_tune --epochs 1
This experiment does not replace the default calibrated scikit-learn serving model.
"""
import argparse
import json
from pathlib import Path
from .config import ROOT
from .data import load
from .taxonomy import LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="distilbert/distilbert-base-uncased")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--output", default=str(ROOT / "models" / "transformer"))
    args = parser.parse_args()
    try:
        import torch
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                                  DataCollatorWithPadding, Trainer, TrainingArguments)
    except ImportError as exc:
        raise SystemExit("Install extra: uv sync --extra transformers") from exc
    train_text, train_labels = load(ROOT / "data/synthetic/train.csv")
    val_text, val_labels = load(ROOT / "data/synthetic/validation.csv")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    l2i = {label: i for i, label in enumerate(LABELS)}
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.tokens = tokenizer(texts, truncation=True, max_length=96)
            self.labels = [l2i[label] for label in labels]
        def __len__(self): return len(self.labels)
        def __getitem__(self, index):
            return {**{k:v[index] for k,v in self.tokens.items()}, "labels": self.labels[index]}
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model,
        num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=l2i)
    cfg = TrainingArguments(output_dir=args.output, num_train_epochs=args.epochs,
        per_device_train_batch_size=8, per_device_eval_batch_size=16,
        eval_strategy="epoch", save_strategy="no", report_to="none", seed=41)
    trainer = Trainer(model=model, args=cfg, train_dataset=Dataset(train_text, train_labels),
        eval_dataset=Dataset(val_text, val_labels), processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer))
    trainer.train(); metrics = trainer.evaluate()
    trainer.save_model(args.output); tokenizer.save_pretrained(args.output)
    Path(args.output, "validation_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
if __name__ == "__main__": main()
