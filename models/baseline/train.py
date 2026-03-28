"""Baseline training script: DistilBERT + linear head, CrossEntropyLoss only.

Same hyperparameters as the middleware training for a fair comparison,
but no MiddleBlock, no SafetyGate, no dual-head loss.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.common import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    TokenizationConfig,
    build_tokenizer,
    choose_device,
    compute_truncation_stats,
    load_json_records,
    set_seed,
    validate_records,
)
from models.baseline.classifier import BaselineClassifier


@dataclass
class BaselineTrainConfig:
    train_data: Path
    val_data: Path
    output_dir: Path
    model_name: str = "distilbert-base-uncased"
    max_length: int = 512
    epochs: int = 6
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    max_train_samples: int = 0
    max_val_samples: int = 0
    pooling: str = "cls_mean"
    early_stopping_patience: int = 3
    dropout: float = 0.2
    gradient_clip_norm: float = 1.0


class PromptDataset(Dataset):
    def __init__(self, records: Sequence[dict]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        row = self.records[index]
        return {
            "prompt_text": row["prompt_text"],
            "label_id": LABEL_TO_ID[row["label"]],
        }


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    return value


def _collate_builder(tokenizer, max_length: int):
    def collate(batch: List[dict]) -> Dict[str, torch.Tensor]:
        texts = [item["prompt_text"] for item in batch]
        labels = torch.tensor([item["label_id"] for item in batch], dtype=torch.long)
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }
    return collate


def _subset(records: List[dict], max_samples: int) -> List[dict]:
    if max_samples and len(records) > max_samples:
        return records[:max_samples]
    return records


def _compute_class_weights(train_records: Sequence[dict], device: torch.device) -> tuple:
    label_ids = np.array([LABEL_TO_ID[row["label"]] for row in train_records])
    class_counts = np.bincount(label_ids, minlength=len(LABEL_TO_ID)).astype(np.float32)
    total = class_counts.sum()
    weights = np.where(class_counts > 0, total / (len(class_counts) * class_counts), 0.0)
    weight_details = {
        "formula": "weight_c = N_total / (K * N_c)",
        "total_samples": int(total),
        "num_classes": len(LABEL_TO_ID),
        "per_class": {
            ID_TO_LABEL[i]: {
                "count": int(class_counts[i]),
                "proportion": round(float(class_counts[i] / total), 4) if total > 0 else 0,
                "weight": round(float(weights[i]), 4),
            }
            for i in range(len(LABEL_TO_ID))
        },
    }
    return torch.tensor(weights, dtype=torch.float32, device=device), weight_details


def _evaluate(model: BaselineClassifier, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    losses: List[float] = []
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(output.logits, labels)
            losses.append(float(loss.item()))

            preds = torch.argmax(output.logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(ID_TO_LABEL.keys()), zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(ID_TO_LABEL.keys()))

    labels_array = np.array(all_labels)
    preds_array = np.array(all_preds)

    # ASR
    jailbreak_idx = LABEL_TO_ID["jailbreak"]
    harmful_idx = LABEL_TO_ID["harmful"]
    jailbreak_mask = labels_array == jailbreak_idx
    harmful_mask = labels_array == harmful_idx

    jailbreak_asr = 0.0
    if jailbreak_mask.sum() > 0:
        jailbreak_asr = float((preds_array[jailbreak_mask] != jailbreak_idx).sum() / jailbreak_mask.sum())
    harmful_asr = 0.0
    if harmful_mask.sum() > 0:
        harmful_asr = float((preds_array[harmful_mask] != harmful_idx).sum() / harmful_mask.sum())
    attack_mask = jailbreak_mask | harmful_mask
    overall_asr = 0.0
    if attack_mask.sum() > 0:
        overall_asr = float((labels_array[attack_mask] != preds_array[attack_mask]).sum() / attack_mask.sum())

    # FRR
    benign_idx = LABEL_TO_ID["benign"]
    benign_mask = labels_array == benign_idx
    frr = 0.0
    if benign_mask.sum() > 0:
        frr = float((preds_array[benign_mask] != benign_idx).sum() / benign_mask.sum())

    return {
        "loss": round(float(np.mean(losses)) if losses else 0.0, 2),
        "accuracy": round(float(accuracy_score(all_labels, all_preds)) if all_labels else 0.0, 2),
        "macro_f1": round(float(np.mean(f1)) if len(f1) else 0.0, 2),
        "asr": {
            "jailbreak_asr": round(jailbreak_asr, 2),
            "harmful_asr": round(harmful_asr, 2),
            "overall_asr": round(overall_asr, 2),
        },
        "frr": round(frr, 4),
        "per_class": {
            ID_TO_LABEL[idx]: {
                "precision": round(float(precision[pos]), 2),
                "recall": round(float(recall[pos]), 2),
                "f1": round(float(f1[pos]), 2),
                "support": int(support[pos]),
            }
            for pos, idx in enumerate(ID_TO_LABEL.keys())
        },
        "confusion_matrix": cm.tolist(),
    }


def run_baseline_training(config: BaselineTrainConfig) -> dict:
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_json_records(config.train_data)
    val_records = load_json_records(config.val_data)
    validate_records(train_records)
    validate_records(val_records)

    train_records = _subset(train_records, config.max_train_samples)
    val_records = _subset(val_records, config.max_val_samples)

    tokenizer = build_tokenizer(config.model_name)
    token_cfg = TokenizationConfig(max_length=config.max_length)
    collate_fn = _collate_builder(tokenizer, token_cfg.max_length)

    # Truncation analysis
    all_texts = [r["prompt_text"] for r in train_records] + [r["prompt_text"] for r in val_records]
    truncation_stats = compute_truncation_stats(all_texts, tokenizer, config.max_length)
    (config.output_dir / "truncation_analysis.json").write_text(
        json.dumps(truncation_stats, indent=2), encoding="utf-8",
    )

    train_loader = DataLoader(
        PromptDataset(train_records), batch_size=config.batch_size,
        shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        PromptDataset(val_records), batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_fn,
    )

    device = choose_device()
    model = BaselineClassifier(
        model_name=config.model_name,
        num_labels=len(LABEL_TO_ID),
        dropout=config.dropout,
        pooling=config.pooling,
    ).to(device)

    class_weights, weight_details = _compute_class_weights(train_records, device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    total_steps = max(1, len(train_loader) * config.epochs)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    training_setup = {
        "model_type": "baseline",
        "model_name": config.model_name,
        "architecture": "Encoder + Dropout + Linear (no MiddleBlock, no SafetyGate)",
        "pooling_strategy": config.pooling,
        "max_token_length": config.max_length,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "warmup_steps": warmup_steps,
        "total_training_steps": total_steps,
        "dropout": config.dropout,
        "gradient_clip_norm": config.gradient_clip_norm,
        "early_stopping_patience": config.early_stopping_patience,
        "optimizer": "AdamW",
        "scheduler": "linear_warmup_linear_decay",
        "loss_function": "CrossEntropyLoss (class-weighted)",
        "seed": config.seed,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "device": str(device),
        "class_weights": weight_details,
        "truncation_analysis": truncation_stats,
    }

    history = []
    best_val_macro_f1 = -1.0
    best_checkpoint = config.output_dir / "best_model.pt"
    patience_counter = 0

    print(f"\n{'='*60}", flush=True)
    print(f"BASELINE Training on {str(device).upper()} | {len(train_records)} train, {len(val_records)} val", flush=True)
    print(f"Model: {config.model_name} | Epochs: {config.epochs} | Batch: {config.batch_size} | LR: {config.learning_rate}", flush=True)
    print(f"Architecture: Encoder \u2192 Dropout \u2192 Linear (no middleware layers)", flush=True)
    print(f"{'='*60}", flush=True)

    for epoch in range(config.epochs):
        model.train()
        train_losses: List[float] = []

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(output.logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)
            optimizer.step()
            scheduler.step()

            train_losses.append(float(loss.item()))

        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        print(f"  Epoch {epoch + 1}/{config.epochs} \u2014 train_loss: {avg_train_loss:.4f} \u2014 evaluating...", flush=True)
        val_metrics = _evaluate(model, val_loader, device)
        print(
            f"  Epoch {epoch + 1}/{config.epochs} \u2014 val_loss: {val_metrics['loss']:.4f} "
            f"| val_acc: {val_metrics['accuracy']:.4f} "
            f"| val_macro_f1: {val_metrics['macro_f1']:.4f} "
            f"| ASR: {val_metrics['asr']['overall_asr']:.4f} "
            f"| FRR: {val_metrics['frr']:.4f}",
            flush=True,
        )
        history.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val": val_metrics})

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            print(f"  \u2713 New best macro-F1: {best_val_macro_f1:.4f} \u2014 saving checkpoint", flush=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": config.model_name,
                    "max_length": config.max_length,
                    "pooling": config.pooling,
                    "label_to_id": LABEL_TO_ID,
                    "id_to_label": ID_TO_LABEL,
                    "model_type": "baseline",
                    "train_config": _json_ready(asdict(config)),
                },
                best_checkpoint,
            )
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1} (patience={config.early_stopping_patience})", flush=True)
                break

    tokenizer_save_error = ""
    try:
        tokenizer.save_pretrained(config.output_dir / "tokenizer")
    except Exception as exc:
        tokenizer_save_error = str(exc)

    outputs = {
        "model_type": "baseline",
        "train_config": _json_ready(asdict(config)),
        "training_setup": _json_ready(training_setup),
        "history": history,
        "best_val_macro_f1": best_val_macro_f1,
        "best_model_path": str(best_checkpoint),
        "early_stopped": patience_counter >= config.early_stopping_patience,
        "epochs_completed": len(history),
        "tokenizer_save_error": tokenizer_save_error,
    }
    (config.output_dir / "training_metrics.json").write_text(
        json.dumps(outputs, indent=2), encoding="utf-8",
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline DistilBERT guardrail classifier")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--pooling", type=str, default="cls_mean", choices=["cls", "mean", "max", "cls_mean"])
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BaselineTrainConfig(
        train_data=args.train_data,
        val_data=args.val_data,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        pooling=args.pooling,
        early_stopping_patience=args.early_stopping_patience,
        dropout=args.dropout,
    )
    result = run_baseline_training(cfg)
    print("Baseline training completed.")
    print(f"Best val macro-F1: {result['best_val_macro_f1']:.4f}")
    print(f"Checkpoint: {result['best_model_path']}")


if __name__ == "__main__":
    main()
