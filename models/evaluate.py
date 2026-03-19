from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.guardrail_classifier import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    GuardrailClassifier,
    build_tokenizer,
    choose_device,
    load_json_records,
    validate_records,
)


class PromptDataset(Dataset):
    def __init__(self, records: Sequence[dict]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        row = self.records[index]
        return {
            "prompt_text": row["prompt_text"],
            "label": row["label"],
            "label_id": LABEL_TO_ID[row["label"]],
        }


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
            "raw": batch,
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }

    return collate


def evaluate_checkpoint(
    checkpoint_path: Path,
    dataset_path: Path,
    output_metrics_path: Path,
    output_samples_path: Path,
    batch_size: int = 16,
    sample_count: int = 12,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)  # trusted local checkpoint

    model_name = checkpoint["model_name"]
    max_length = int(checkpoint.get("max_length", 192))

    records = load_json_records(dataset_path)
    validate_records(records)

    tokenizer = build_tokenizer(model_name)
    loader = DataLoader(
        PromptDataset(records),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_builder(tokenizer, max_length),
    )

    device = choose_device()
    model = GuardrailClassifier(model_name=model_name, num_labels=len(LABEL_TO_ID))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_preds: List[int] = []
    all_probs: List[List[float]] = []
    all_labels: List[int] = []
    sample_rows = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            for i, row in enumerate(batch["raw"]):
                if len(sample_rows) >= sample_count:
                    break
                pred_idx = int(preds[i].item())
                sample_rows.append(
                    {
                        "prompt_text": row["prompt_text"],
                        "true_label": row["label"],
                        "pred_label": ID_TO_LABEL[pred_idx],
                        "probabilities": {
                            ID_TO_LABEL[j]: round(float(all_probs[-len(batch["raw"]) + i][j]), 6)
                            for j in range(len(ID_TO_LABEL))
                        },
                    }
                )

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=list(ID_TO_LABEL.keys()),
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(ID_TO_LABEL.keys()))

    # Calculate ASR (Attack Success Rate) metrics
    # ASR = ratio of attack prompts (jailbreak + harmful) that were NOT detected (predicted as benign)
    jailbreak_idx = LABEL_TO_ID["jailbreak"]
    harmful_idx = LABEL_TO_ID["harmful"]
    benign_idx = LABEL_TO_ID["benign"]
    
    labels_array = np.array(all_labels)
    preds_array = np.array(all_preds)
    
    # Per-class ASR: for each attack class, ratio of misdetections
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
    
    # Overall ASR: ratio of all attack prompts that were misclassified (not correctly detected)
    attack_mask = jailbreak_mask | harmful_mask
    overall_asr = 0.0
    if attack_mask.sum() > 0:
        attack_misclassified = (
            labels_array[attack_mask] != preds_array[attack_mask]
        ).sum()
        overall_asr = float(attack_misclassified / attack_mask.sum())

    metrics = {
        "dataset": str(dataset_path),
        "num_examples": len(records),
        "accuracy": round(float(accuracy_score(all_labels, all_preds)) if all_labels else 0.0, 2),
        "macro_f1": round(float(np.mean(f1)) if len(f1) else 0.0, 2),
        "asr": {
            "jailbreak_asr": round(jailbreak_asr, 2),
            "harmful_asr": round(harmful_asr, 2),
            "overall_asr": round(overall_asr, 2),
            "description": "Attack Success Rate: ratio of attack prompts incorrectly classified as benign or wrong class"
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

    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    output_samples_path.parent.mkdir(parents=True, exist_ok=True)

    output_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    output_samples_path.write_text(json.dumps(sample_rows, indent=2, ensure_ascii=True), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DistilBERT guardrail classifier")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    parser.add_argument("--output-samples", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-count", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        output_metrics_path=args.output_metrics,
        output_samples_path=args.output_samples,
        batch_size=args.batch_size,
        sample_count=args.sample_count,
    )
    print("Evaluation completed.")
    print(f"Examples: {metrics['num_examples']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
