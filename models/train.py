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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.guardrail_classifier import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    GuardrailClassifier,
    TokenizationConfig,
    build_tokenizer,
    choose_device,
    load_json_records,
    set_seed,
    validate_records,
)


@dataclass
class TrainConfig:
    train_data: Path
    val_data: Path
    output_dir: Path
    model_name: str = "distilbert-base-uncased"
    max_length: int = 192
    epochs: int = 2
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    max_train_samples: int = 0
    max_val_samples: int = 0
    smoke_mode: bool = False


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


def _compute_class_weights(train_records: Sequence[dict], device: torch.device) -> torch.Tensor:
    label_ids = np.array([LABEL_TO_ID[row["label"]] for row in train_records])
    class_counts = np.bincount(label_ids, minlength=len(LABEL_TO_ID)).astype(np.float32)
    total = class_counts.sum()
    weights = np.where(class_counts > 0, total / (len(class_counts) * class_counts), 0.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _evaluate(model: GuardrailClassifier, loader: DataLoader, device: torch.device) -> dict:
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

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            losses.append(float(loss.item()))

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=list(ID_TO_LABEL.keys()),
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(ID_TO_LABEL.keys()))

    # Calculate ASR (Attack Success Rate) metrics
    jailbreak_idx = LABEL_TO_ID["jailbreak"]
    harmful_idx = LABEL_TO_ID["harmful"]
    
    labels_array = np.array(all_labels)
    preds_array = np.array(all_preds)
    
    jailbreak_mask = labels_array == jailbreak_idx
    harmful_mask = labels_array == harmful_idx
    
    jailbreak_asr = 0.0
    if jailbreak_mask.sum() > 0:
        jailbreak_missed = (
            (labels_array[jailbreak_mask] == jailbreak_idx) & 
            (preds_array[jailbreak_mask] != jailbreak_idx)
        ).sum()
        jailbreak_asr = float(jailbreak_missed / jailbreak_mask.sum())
    
    harmful_asr = 0.0
    if harmful_mask.sum() > 0:
        harmful_missed = (
            (labels_array[harmful_mask] == harmful_idx) & 
            (preds_array[harmful_mask] != harmful_idx)
        ).sum()
        harmful_asr = float(harmful_missed / harmful_mask.sum())
    
    attack_mask = jailbreak_mask | harmful_mask
    overall_asr = 0.0
    if attack_mask.sum() > 0:
        attack_misclassified = (
            labels_array[attack_mask] != preds_array[attack_mask]
        ).sum()
        overall_asr = float(attack_misclassified / attack_mask.sum())

    return {
        "loss": round(float(np.mean(losses)) if losses else 0.0, 2),
        "accuracy": round(float(accuracy_score(all_labels, all_preds)) if all_labels else 0.0, 2),
        "macro_f1": round(float(np.mean(f1)) if len(f1) else 0.0, 2),
        "asr": {
            "jailbreak_asr": round(jailbreak_asr, 2),
            "harmful_asr": round(harmful_asr, 2),
            "overall_asr": round(overall_asr, 2),
        },
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


def run_training(config: TrainConfig) -> dict:
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

    train_loader = DataLoader(
        PromptDataset(train_records),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        PromptDataset(val_records),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    device = choose_device()
    model = GuardrailClassifier(model_name=config.model_name, num_labels=len(LABEL_TO_ID)).to(device)

    class_weights = _compute_class_weights(train_records, device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    total_steps = max(1, len(train_loader) * config.epochs)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history = []
    best_val_macro_f1 = -1.0
    best_checkpoint = config.output_dir / "best_model.pt"

    for epoch in range(config.epochs):
        model.train()
        train_losses: List[float] = []

        for batch in train_loader:
            if not config.smoke_mode:
                optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            if not config.smoke_mode:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            train_losses.append(float(loss.item()))

        val_metrics = _evaluate(model, val_loader, device)
        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "val": val_metrics,
        }
        history.append(epoch_result)

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": config.model_name,
                    "max_length": config.max_length,
                    "label_to_id": LABEL_TO_ID,
                    "id_to_label": ID_TO_LABEL,
                    "train_config": asdict(config),
                },
                best_checkpoint,
            )

    tokenizer_save_error = ""
    try:
        tokenizer.save_pretrained(config.output_dir / "tokenizer")
    except Exception as exc:  # Non-fatal for subset pipeline verification artifacts.
        tokenizer_save_error = str(exc)

    outputs = {
        "train_config": _json_ready(asdict(config)),
        "smoke_mode": config.smoke_mode,
        "history": history,
        "best_val_macro_f1": best_val_macro_f1,
        "best_model_path": str(best_checkpoint),
        "tokenizer_save_error": tokenizer_save_error,
    }
    (config.output_dir / "training_metrics.json").write_text(
        json.dumps(outputs, indent=2),
        encoding="utf-8",
    )

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DistilBERT guardrail classifier")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--smoke-mode", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
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
        smoke_mode=args.smoke_mode,
    )
    result = run_training(cfg)
    print("Training completed.")
    print(f"Best val macro-F1: {result['best_val_macro_f1']:.4f}")
    print(f"Checkpoint: {result['best_model_path']}")


if __name__ == "__main__":
    main()
