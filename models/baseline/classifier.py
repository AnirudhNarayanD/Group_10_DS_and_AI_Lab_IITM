"""Baseline Guardrail Classifier: Encoder + Dropout + Linear head.

No MiddleBlock, no ClassificationDecoder, no SafetyGate.
This serves as the performance baseline to measure the value-add of the
middleware architecture layers.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers import AutoModel

from models.common import (
    POOLING_STRATEGIES,
    GuardrailModelOutput,
    pool_encoder_output,
)


class BaselineClassifier(nn.Module):
    """Plain DistilBERT fine-tuned with a single linear classification head.

    Architecture:
        Encoder (DistilBERT) → Pooling → Dropout → nn.Linear → 3-class logits

    The safety_score in the output is always zeros (no safety gate).
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int = 3,
        dropout: float = 0.2,
        pooling: str = "cls_mean",
    ) -> None:
        super().__init__()
        if pooling not in POOLING_STRATEGIES:
            raise ValueError(f"pooling must be one of {POOLING_STRATEGIES}, got '{pooling}'")
        self.pooling = pooling

        # Encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        pool_output_size = hidden_size * 2 if pooling == "cls_mean" else hidden_size

        # Single linear head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(pool_output_size, num_labels)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> GuardrailModelOutput:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = pool_encoder_output(outputs.last_hidden_state, attention_mask, self.pooling)
        logits = self.classifier(self.dropout(pooled))

        safety_score = torch.zeros(input_ids.shape[0], 1, device=input_ids.device)
        return GuardrailModelOutput(logits=logits, safety_score=safety_score)
