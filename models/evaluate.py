from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.common import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    ThresholdConfig,
    build_tokenizer,
    choose_device,
    compute_truncation_stats,
    load_json_records,
    validate_records,
)
from models.middleware.classifier import ArchConfig, GuardrailClassifier


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


def _compute_roc_pr_curves(labels_array: np.ndarray, probs_array: np.ndarray) -> dict:
    """Compute per-class ROC and PR curves with AUC values."""
    n_classes = len(LABEL_TO_ID)
    curves = {}
    for class_idx in range(n_classes):
        class_name = ID_TO_LABEL[class_idx]
        binary_labels = (labels_array == class_idx).astype(int)
        class_probs = probs_array[:, class_idx]

        # ROC curve
        try:
            fpr, tpr, roc_thresholds = roc_curve(binary_labels, class_probs)
            roc_auc_val = auc(fpr, tpr)
        except ValueError:
            fpr, tpr, roc_thresholds = np.array([0]), np.array([0]), np.array([0.5])
            roc_auc_val = 0.0

        # PR curve
        try:
            prec_curve, rec_curve, pr_thresholds = precision_recall_curve(binary_labels, class_probs)
            pr_auc_val = auc(rec_curve, prec_curve)
        except ValueError:
            prec_curve, rec_curve, pr_thresholds = np.array([0]), np.array([0]), np.array([0.5])
            pr_auc_val = 0.0

        # Find optimal threshold (Youden's J statistic for ROC)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = float(roc_thresholds[optimal_idx]) if len(roc_thresholds) > optimal_idx else 0.5

        # Sample curve points for JSON serialization (avoid huge arrays)
        max_points = 50
        roc_indices = np.linspace(0, len(fpr) - 1, min(max_points, len(fpr)), dtype=int)
        pr_indices = np.linspace(0, len(prec_curve) - 1, min(max_points, len(prec_curve)), dtype=int)

        curves[class_name] = {
            "roc_auc": round(float(roc_auc_val), 4),
            "pr_auc": round(float(pr_auc_val), 4),
            "optimal_threshold": round(optimal_threshold, 4),
            "roc_curve": {
                "fpr": [round(float(fpr[i]), 4) for i in roc_indices],
                "tpr": [round(float(tpr[i]), 4) for i in roc_indices],
            },
            "pr_curve": {
                "precision": [round(float(prec_curve[i]), 4) for i in pr_indices],
                "recall": [round(float(rec_curve[i]), 4) for i in pr_indices],
            },
        }

    # Macro-average ROC-AUC
    try:
        macro_roc_auc = roc_auc_score(
            labels_array, probs_array, multi_class="ovr", average="macro"
        )
    except ValueError:
        macro_roc_auc = 0.0

    curves["macro_roc_auc"] = round(float(macro_roc_auc), 4)
    return curves


def _compute_calibration(labels_array: np.ndarray, probs_array: np.ndarray, n_bins: int = 10) -> dict:
    """Compute Expected Calibration Error (ECE) and reliability diagram data."""
    n_classes = len(LABEL_TO_ID)
    calibration = {}
    overall_ece = 0.0
    total_weight = 0.0

    for class_idx in range(n_classes):
        class_name = ID_TO_LABEL[class_idx]
        binary_labels = (labels_array == class_idx).astype(int)
        class_probs = probs_array[:, class_idx]

        # Compute ECE for this class
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_ece = 0.0
        bin_details = []

        for b in range(n_bins):
            mask = (class_probs >= bin_boundaries[b]) & (class_probs < bin_boundaries[b + 1])
            if b == n_bins - 1:
                mask = (class_probs >= bin_boundaries[b]) & (class_probs <= bin_boundaries[b + 1])
            n_in_bin = mask.sum()
            if n_in_bin == 0:
                continue
            avg_confidence = float(class_probs[mask].mean())
            avg_accuracy = float(binary_labels[mask].mean())
            bin_ece += abs(avg_confidence - avg_accuracy) * n_in_bin
            bin_details.append({
                "bin": f"[{bin_boundaries[b]:.2f}, {bin_boundaries[b+1]:.2f})",
                "count": int(n_in_bin),
                "avg_confidence": round(avg_confidence, 4),
                "avg_accuracy": round(avg_accuracy, 4),
                "gap": round(abs(avg_confidence - avg_accuracy), 4),
            })

        class_ece = bin_ece / len(labels_array) if len(labels_array) > 0 else 0.0
        overall_ece += class_ece
        total_weight += 1.0

        # Reliability diagram data
        try:
            fraction_pos, mean_predicted = calibration_curve(binary_labels, class_probs, n_bins=n_bins, strategy="uniform")
            reliability = {
                "fraction_of_positives": [round(float(x), 4) for x in fraction_pos],
                "mean_predicted_value": [round(float(x), 4) for x in mean_predicted],
            }
        except ValueError:
            reliability = {"fraction_of_positives": [], "mean_predicted_value": []}

        calibration[class_name] = {
            "ece": round(float(class_ece), 4),
            "bin_details": bin_details,
            "reliability_diagram": reliability,
        }

    calibration["overall_ece"] = round(float(overall_ece / total_weight) if total_weight > 0 else 0.0, 4)
    calibration["interpretation"] = (
        "ECE (Expected Calibration Error) measures how well predicted probabilities match actual "
        "frequencies. ECE < 0.05 indicates good calibration. Values > 0.1 suggest the model is "
        "over-confident or under-confident and may benefit from temperature scaling."
    )
    return calibration


def _compute_error_analysis(
    records: List[dict],
    all_labels: List[int],
    all_preds: List[int],
    all_probs: List[List[float]],
) -> dict:
    """Systematic error analysis: FP/FN breakdown, misclassification patterns."""
    labels_array = np.array(all_labels)
    preds_array = np.array(all_preds)

    error_analysis = {
        "total_samples": len(all_labels),
        "correct": int((labels_array == preds_array).sum()),
        "incorrect": int((labels_array != preds_array).sum()),
        "error_rate": round(float((labels_array != preds_array).mean()), 4) if len(all_labels) > 0 else 0.0,
    }

    # Per-class FP/FN analysis
    per_class_errors = {}
    for class_idx in range(len(LABEL_TO_ID)):
        class_name = ID_TO_LABEL[class_idx]
        tp = int(((labels_array == class_idx) & (preds_array == class_idx)).sum())
        fp = int(((labels_array != class_idx) & (preds_array == class_idx)).sum())
        fn = int(((labels_array == class_idx) & (preds_array != class_idx)).sum())
        tn = int(((labels_array != class_idx) & (preds_array != class_idx)).sum())

        # Collect FP and FN examples
        fp_examples = []
        fn_examples = []
        for i in range(len(all_labels)):
            if all_labels[i] != class_idx and all_preds[i] == class_idx and len(fp_examples) < 5:
                fp_examples.append({
                    "text_preview": records[i]["prompt_text"][:120],
                    "true_label": ID_TO_LABEL[all_labels[i]],
                    "confidence": round(all_probs[i][class_idx], 4),
                })
            if all_labels[i] == class_idx and all_preds[i] != class_idx and len(fn_examples) < 5:
                fn_examples.append({
                    "text_preview": records[i]["prompt_text"][:120],
                    "pred_label": ID_TO_LABEL[all_preds[i]],
                    "confidence": round(all_probs[i][all_preds[i]], 4),
                })

        per_class_errors[class_name] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "fp_examples": fp_examples,
            "fn_examples": fn_examples,
        }

    error_analysis["per_class_errors"] = per_class_errors

    # Misclassification matrix (where do errors go?)
    misclass_matrix = {}
    for true_idx in range(len(LABEL_TO_ID)):
        for pred_idx in range(len(LABEL_TO_ID)):
            if true_idx != pred_idx:
                count = int(((labels_array == true_idx) & (preds_array == pred_idx)).sum())
                if count > 0:
                    key = f"{ID_TO_LABEL[true_idx]}_predicted_as_{ID_TO_LABEL[pred_idx]}"
                    misclass_matrix[key] = count
    error_analysis["misclassification_flows"] = misclass_matrix

    # Confidence on errors vs correct predictions
    correct_mask = labels_array == preds_array
    if correct_mask.any():
        correct_confs = [max(all_probs[i]) for i in range(len(all_probs)) if correct_mask[i]]
        error_analysis["correct_avg_confidence"] = round(float(np.mean(correct_confs)), 4)
    if (~correct_mask).any():
        error_confs = [max(all_probs[i]) for i in range(len(all_probs)) if not correct_mask[i]]
        error_analysis["error_avg_confidence"] = round(float(np.mean(error_confs)), 4)

    return error_analysis


def _compute_per_attack_type(
    records: List[dict],
    all_labels: List[int],
    all_preds: List[int],
    all_probs: List[List[float]],
) -> dict:
    """Per-attack-type performance breakdown for jailbreak taxonomy."""
    attack_groups: Dict[str, List[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        attack_type = rec.get("attack_type", "none")
        attack_groups[attack_type].append(i)

    results = {}
    for attack_type, indices in sorted(attack_groups.items()):
        group_labels = [all_labels[i] for i in indices]
        group_preds = [all_preds[i] for i in indices]

        if not group_labels:
            continue

        group_labels_arr = np.array(group_labels)
        group_preds_arr = np.array(group_preds)

        correct = int((group_labels_arr == group_preds_arr).sum())
        total = len(group_labels)
        accuracy = round(correct / total, 4) if total > 0 else 0.0

        # ASR for attack types (not benign)
        attack_asr = 0.0
        attack_count = 0
        for i in indices:
            if all_labels[i] != LABEL_TO_ID["benign"]:
                attack_count += 1
                if all_labels[i] != all_preds[i]:
                    attack_asr += 1
        if attack_count > 0:
            attack_asr = round(attack_asr / attack_count, 4)

        # Confusion for this group
        label_dist = Counter(ID_TO_LABEL[l] for l in group_labels)
        pred_dist = Counter(ID_TO_LABEL[p] for p in group_preds)

        results[attack_type] = {
            "total_samples": total,
            "accuracy": accuracy,
            "attack_success_rate": attack_asr if attack_type != "none" else "N/A",
            "true_label_distribution": dict(label_dist),
            "predicted_distribution": dict(pred_dist),
        }

    return results


def _compute_threshold_sweep(
    labels_array: np.ndarray,
    probs_array: np.ndarray,
) -> dict:
    """Sweep thresholds for ASR vs FRR (False Rejection Rate) trade-off analysis."""
    thresholds = np.arange(0.1, 1.0, 0.05)
    sweep_results = []

    benign_idx = LABEL_TO_ID["benign"]
    attack_mask = labels_array != benign_idx
    benign_mask = labels_array == benign_idx

    for thresh in thresholds:
        # Block if max attack probability > threshold
        attack_probs = np.max(probs_array[:, 1:], axis=1)  # max of jailbreak, harmful
        pred_block = attack_probs >= thresh

        # ASR: attacks that are NOT blocked
        if attack_mask.sum() > 0:
            asr = float((~pred_block[attack_mask]).sum() / attack_mask.sum())
        else:
            asr = 0.0

        # FRR: benign prompts that ARE blocked (false rejections)
        if benign_mask.sum() > 0:
            frr = float(pred_block[benign_mask].sum() / benign_mask.sum())
        else:
            frr = 0.0

        sweep_results.append({
            "threshold": round(float(thresh), 2),
            "asr": round(asr, 4),
            "frr": round(frr, 4),
        })

    return {
        "description": "ASR (Attack Success Rate) vs FRR (False Rejection Rate) at various block thresholds",
        "sweep": sweep_results,
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    dataset_path: Path,
    output_metrics_path: Path,
    output_samples_path: Path,
    batch_size: int = 16,
    sample_count: int = 20,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)  # trusted local checkpoint

    model_name = checkpoint["model_name"]
    max_length = int(checkpoint.get("max_length", 512))
    pooling = checkpoint.get("pooling", "cls_mean")

    records = load_json_records(dataset_path)
    validate_records(records)

    tokenizer = build_tokenizer(model_name)

    # --- Truncation impact analysis ---
    all_texts = [r["prompt_text"] for r in records]
    truncation_stats = compute_truncation_stats(all_texts, tokenizer, max_length)

    loader = DataLoader(
        PromptDataset(records),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_builder(tokenizer, max_length),
    )

    device = choose_device()
    arch_config_dict = checkpoint.get("arch_config", None)
    arch_cfg = ArchConfig(**arch_config_dict) if arch_config_dict else ArchConfig()
    model = GuardrailClassifier(
        model_name=model_name,
        num_labels=len(LABEL_TO_ID),
        pooling=pooling,
        arch_config=arch_cfg,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_preds: List[int] = []
    all_probs: List[List[float]] = []
    all_labels: List[int] = []
    all_safety_scores: List[float] = []
    sample_rows = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(output.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_safety_scores.extend(output.safety_score.squeeze(-1).cpu().tolist())

            for i, row in enumerate(batch["raw"]):
                if len(sample_rows) >= sample_count:
                    break
                pred_idx = int(preds[i].item())
                sample_rows.append(
                    {
                        "prompt_text": row["prompt_text"],
                        "true_label": row["label"],
                        "pred_label": ID_TO_LABEL[pred_idx],
                        "correct": row["label"] == ID_TO_LABEL[pred_idx],
                        "probabilities": {
                            ID_TO_LABEL[j]: round(float(probs[i][j].item()), 6)
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

    # ASR metrics
    labels_array = np.array(all_labels)
    preds_array = np.array(all_preds)
    probs_array = np.array(all_probs)

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

    # --- Comprehensive evaluation metrics ---
    roc_pr_curves = _compute_roc_pr_curves(labels_array, probs_array)
    calibration = _compute_calibration(labels_array, probs_array)
    error_analysis = _compute_error_analysis(records, all_labels, all_preds, all_probs)
    per_attack = _compute_per_attack_type(records, all_labels, all_preds, all_probs)
    threshold_sweep = _compute_threshold_sweep(labels_array, probs_array)

    metrics = {
        "dataset": str(dataset_path),
        "num_examples": len(records),
        "pooling_strategy": pooling,
        "max_token_length": max_length,
        "accuracy": round(float(accuracy_score(all_labels, all_preds)) if all_labels else 0.0, 4),
        "macro_f1": round(float(np.mean(f1)) if len(f1) else 0.0, 4),
        "asr": {
            "jailbreak_asr": round(jailbreak_asr, 4),
            "harmful_asr": round(harmful_asr, 4),
            "overall_asr": round(overall_asr, 4),
            "description": "Attack Success Rate: ratio of attack prompts incorrectly classified",
        },
        "per_class": {
            ID_TO_LABEL[idx]: {
                "precision": round(float(precision[pos]), 4),
                "recall": round(float(recall[pos]), 4),
                "f1": round(float(f1[pos]), 4),
                "support": int(support[pos]),
            }
            for pos, idx in enumerate(ID_TO_LABEL.keys())
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [ID_TO_LABEL[i] for i in range(len(ID_TO_LABEL))],
        "roc_pr_curves": roc_pr_curves,
        "confidence_calibration": calibration,
        "error_analysis": error_analysis,
        "per_attack_type_performance": per_attack,
        "threshold_sweep": threshold_sweep,
        "truncation_analysis": truncation_stats,
        "safety_gate": {
            "avg_score": round(float(np.mean(all_safety_scores)), 4) if all_safety_scores else 0.0,
            "avg_score_benign": round(
                float(np.mean([s for s, l in zip(all_safety_scores, all_labels) if l == LABEL_TO_ID["benign"]])), 4
            ) if any(l == LABEL_TO_ID["benign"] for l in all_labels) else 0.0,
            "avg_score_attack": round(
                float(np.mean([s for s, l in zip(all_safety_scores, all_labels) if l != LABEL_TO_ID["benign"]])), 4
            ) if any(l != LABEL_TO_ID["benign"] for l in all_labels) else 0.0,
        },
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
    parser.add_argument("--sample-count", type=int, default=20)
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
