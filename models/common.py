"""Shared utilities, constants, and pooling strategies used by both
baseline and middleware classifiers.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Label mappings (shared across all models)
# ---------------------------------------------------------------------------

LABEL_TO_ID: Dict[str, int] = {
    "benign": 0,
    "jailbreak": 1,
    "harmful": 2,
}
ID_TO_LABEL: Dict[int, str] = {idx: label for label, idx in LABEL_TO_ID.items()}

POOLING_STRATEGIES = ("cls", "mean", "max", "cls_mean")


# ---------------------------------------------------------------------------
# Model output container
# ---------------------------------------------------------------------------

@dataclass
class GuardrailModelOutput:
    """Output from any guardrail classifier (baseline or middleware)."""
    logits: torch.Tensor           # [B, num_labels] classification logits
    safety_score: torch.Tensor     # [B, 1] safety gate score (0=safe, 1=unsafe)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TokenizationConfig:
    max_length: int = 512


@dataclass
class ThresholdConfig:
    """Decision threshold strategy for mapping probabilities to allow/block."""
    benign_threshold: float = 0.5
    jailbreak_threshold: float = 0.5
    harmful_threshold: float = 0.5
    block_threshold: float = 0.5
    strategy: str = "argmax"

    def apply(self, probs: torch.Tensor) -> torch.Tensor:
        if self.strategy == "argmax":
            return torch.argmax(probs, dim=-1)

        batch_size = probs.shape[0]
        preds = torch.zeros(batch_size, dtype=torch.long, device=probs.device)
        thresholds = torch.tensor(
            [self.benign_threshold, self.jailbreak_threshold, self.harmful_threshold],
            device=probs.device,
        )

        for i in range(batch_size):
            p = probs[i]
            if self.strategy == "safety_biased":
                attack_max = max(p[LABEL_TO_ID["jailbreak"]].item(), p[LABEL_TO_ID["harmful"]].item())
                if attack_max >= self.block_threshold:
                    preds[i] = LABEL_TO_ID["jailbreak"] if p[1] >= p[2] else LABEL_TO_ID["harmful"]
                else:
                    preds[i] = LABEL_TO_ID["benign"]
            else:
                above = p >= thresholds
                if above.any():
                    candidates = p.clone()
                    candidates[~above] = -1.0
                    preds[i] = torch.argmax(candidates)
                else:
                    preds[i] = LABEL_TO_ID["benign"]
        return preds


# ---------------------------------------------------------------------------
# Pooling strategies
# ---------------------------------------------------------------------------

def _mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


def _max_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    hidden_states = hidden_states.clone()
    hidden_states[mask_expanded == 0] = -1e9
    return torch.max(hidden_states, dim=1).values


def pool_encoder_output(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    strategy: str,
) -> torch.Tensor:
    """Apply a named pooling strategy to encoder hidden states."""
    if strategy == "cls":
        return hidden_states[:, 0, :]
    elif strategy == "mean":
        return _mean_pooling(hidden_states, attention_mask)
    elif strategy == "max":
        return _max_pooling(hidden_states, attention_mask)
    elif strategy == "cls_mean":
        cls_emb = hidden_states[:, 0, :]
        mean_emb = _mean_pooling(hidden_states, attention_mask)
        return torch.cat([cls_emb, mean_emb], dim=-1)
    raise ValueError(f"Unknown pooling: {strategy}")


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_text_for_model(text: str) -> str:
    """Light normalization that preserves adversarial formatting signals."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{3,}", "  ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Truncation analysis
# ---------------------------------------------------------------------------

def compute_truncation_stats(texts: List[str], tokenizer, max_length: int) -> dict:
    total = len(texts)
    if total == 0:
        return {"total": 0, "truncated": 0, "truncation_rate": 0.0}

    lengths = []
    truncated_count = 0
    truncated_lengths = []

    for text in texts:
        tokens = tokenizer(text, add_special_tokens=True, truncation=False)
        token_len = len(tokens["input_ids"])
        lengths.append(token_len)
        if token_len > max_length:
            truncated_count += 1
            truncated_lengths.append(token_len)

    lengths_arr = np.array(lengths)
    return {
        "total": total,
        "max_length_setting": max_length,
        "truncated": truncated_count,
        "truncation_rate": round(truncated_count / total, 4),
        "token_length_stats": {
            "min": int(lengths_arr.min()),
            "max": int(lengths_arr.max()),
            "mean": round(float(lengths_arr.mean()), 1),
            "median": round(float(np.median(lengths_arr)), 1),
            "p90": round(float(np.percentile(lengths_arr, 90)), 1),
            "p95": round(float(np.percentile(lengths_arr, 95)), 1),
            "p99": round(float(np.percentile(lengths_arr, 99)), 1),
        },
        "truncated_token_lengths": {
            "min": int(min(truncated_lengths)) if truncated_lengths else 0,
            "max": int(max(truncated_lengths)) if truncated_lengths else 0,
            "mean": round(float(np.mean(truncated_lengths)), 1) if truncated_lengths else 0,
        },
    }


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json_records(path: Path) -> List[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list records in {path}")
    return [row for row in payload if isinstance(row, dict)]


def validate_records(records: Iterable[dict]) -> None:
    required = {"prompt_text", "label", "split"}
    for idx, row in enumerate(records):
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"Record {idx} missing fields: {missing}")
        if row["label"] not in LABEL_TO_ID:
            raise ValueError(f"Record {idx} has unknown label: {row['label']}")


def build_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
