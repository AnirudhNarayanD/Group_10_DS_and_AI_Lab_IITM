from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

LABEL_TO_ID: Dict[str, int] = {
    "benign": 0,
    "jailbreak": 1,
    "harmful": 2,
}
ID_TO_LABEL: Dict[int, str] = {idx: label for label, idx in LABEL_TO_ID.items()}


@dataclass
class TokenizationConfig:
    max_length: int = 192


class GuardrailClassifier(nn.Module):
    """DistilBERT encoder with a linear classification head for 3-way labels."""

    def __init__(self, model_name: str, num_labels: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT has no pooler output; CLS token embedding is used as sentence representation.
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_embedding))
        return logits


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
