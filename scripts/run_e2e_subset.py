from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.evaluate import evaluate_checkpoint
from models.train import TrainConfig, run_training


def read_json(path: Path) -> List[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}")
    return [row for row in payload if isinstance(row, dict)]


def stratified_subset(records: List[dict], target_size: int, seed: int) -> List[dict]:
    if target_size <= 0 or target_size >= len(records):
        return list(records)

    rng = random.Random(seed)
    by_label: Dict[str, List[dict]] = defaultdict(list)
    for row in records:
        by_label[row["label"]].append(row)

    total = len(records)
    out = []
    for label, rows in by_label.items():
        rng.shuffle(rows)
        n = max(1, round(target_size * (len(rows) / total)))
        out.extend(rows[:n])

    if len(out) > target_size:
        rng.shuffle(out)
        out = out[:target_size]

    if len(out) < target_size:
        used = {row.get("prompt_id", f"idx_{i}") for i, row in enumerate(out)}
        leftovers = [
            row for i, row in enumerate(records)
            if row.get("prompt_id", f"idx_{i}") not in used
        ]
        rng.shuffle(leftovers)
        out.extend(leftovers[: target_size - len(out)])

    return out


def write_subset(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=True), encoding="utf-8")


def to_json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: to_json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_json_ready(v) for v in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fast end-to-end subset pipeline for Milestone 3")
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("datasets/dataset_outputs/evaluation_dataset/splits"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("milestone_3_outputs"))
    parser.add_argument("--run-name", type=str, default="subset_e2e")
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-size", type=int, default=210)
    parser.add_argument("--val-size", type=int, default=60)
    parser.add_argument("--test-size", type=int, default=60)
    parser.add_argument("--smoke-mode", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_full = read_json(args.splits_dir / "train.json")
    val_full = read_json(args.splits_dir / "validation.json")
    test_full = read_json(args.splits_dir / "test.json")

    train_subset = stratified_subset(train_full, args.train_size, seed=args.seed)
    val_subset = stratified_subset(val_full, args.val_size, seed=args.seed + 1)
    test_subset = stratified_subset(test_full, args.test_size, seed=args.seed + 2)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"{args.run_name}_{timestamp}"
    data_dir = run_dir / "subset_data"
    model_dir = run_dir / "model"
    eval_dir = run_dir / "evaluation"

    train_path = data_dir / "train_subset.json"
    val_path = data_dir / "validation_subset.json"
    test_path = data_dir / "test_subset.json"

    write_subset(train_path, train_subset)
    write_subset(val_path, val_subset)
    write_subset(test_path, test_subset)

    train_cfg = TrainConfig(
        train_data=train_path,
        val_data=val_path,
        output_dir=model_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        smoke_mode=args.smoke_mode,
    )
    try:
        train_result = run_training(train_cfg)

        best_checkpoint = Path(train_result["best_model_path"])
        val_metrics = evaluate_checkpoint(
            checkpoint_path=best_checkpoint,
            dataset_path=val_path,
            output_metrics_path=eval_dir / "validation_metrics.json",
            output_samples_path=eval_dir / "validation_samples.json",
            batch_size=args.batch_size,
        )
        test_metrics = evaluate_checkpoint(
            checkpoint_path=best_checkpoint,
            dataset_path=test_path,
            output_metrics_path=eval_dir / "test_metrics.json",
            output_samples_path=eval_dir / "test_samples.json",
            batch_size=args.batch_size,
        )

        run_summary = {
            "run_dir": str(run_dir),
            "created_utc": datetime.now(UTC).isoformat(),
            "subset_sizes": {
                "train": len(train_subset),
                "validation": len(val_subset),
                "test": len(test_subset),
            },
            "train_config": to_json_ready(asdict(train_cfg)),
            "best_checkpoint": str(best_checkpoint),
            "best_val_macro_f1": train_result["best_val_macro_f1"],
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "loss_function": "CrossEntropyLoss(class_weighted)",
            "optimizer": "AdamW",
            "smoke_mode": args.smoke_mode,
        }
        (run_dir / "run_summary.json").write_text(
            json.dumps(run_summary, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

        print("Milestone 3 subset E2E pipeline completed.")
        print(f"Run directory: {run_dir}")
        print(f"Best checkpoint: {best_checkpoint}")
        print(f"Validation macro-F1: {val_metrics['macro_f1']:.4f}")
        print(f"Test macro-F1: {test_metrics['macro_f1']:.4f}")
    except Exception:
        (run_dir / "run_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
