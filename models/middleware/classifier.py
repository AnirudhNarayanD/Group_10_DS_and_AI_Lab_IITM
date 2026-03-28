"""Middleware Guardrail Classifier: Encoder → MiddleBlock → Decoder + SafetyGate.

Full guardrail architecture with:
  - MiddleBlock: Dense-LN-GELU feature refinement with residual connection
  - ClassificationDecoder: Multi-layer classification head (→ 3-class logits)
  - SafetyGate: Binary unsafe detector (→ safety score [0,1])
  - Rule-based pre-filter (regex jailbreak patterns + harmful keywords)
  - Multi-turn escalation detection (cross-turn regex sequences)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import nn
from transformers import AutoModel

from models.common import (
    LABEL_TO_ID,
    POOLING_STRATEGIES,
    GuardrailModelOutput,
    pool_encoder_output,
)


# ---------------------------------------------------------------------------
# Architecture config
# ---------------------------------------------------------------------------

@dataclass
class ArchConfig:
    """Architecture configuration for the full guardrail classifier."""
    intermediate_size: int = 512
    decoder_hidden_size: int = 256
    safety_gate_hidden: int = 128
    enable_safety_gate: bool = True


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class MiddleBlock(nn.Module):
    """Feature-refinement block between encoder pooling and classification head.

    Architecture: Dense → LayerNorm → GELU → Dropout → Dense → LayerNorm → GELU → Dropout
    with a residual connection across the block.
    """

    def __init__(self, input_size: int, intermediate_size: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, intermediate_size)
        self.ln1 = nn.LayerNorm(intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, intermediate_size)
        self.ln2 = nn.LayerNorm(intermediate_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual_proj = (
            nn.Linear(input_size, intermediate_size)
            if input_size != intermediate_size
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        out = self.dropout(self.act(self.ln1(self.fc1(x))))
        out = self.dropout(self.act(self.ln2(self.fc2(out))))
        return out + residual


class ClassificationDecoder(nn.Module):
    """Multi-layer classification head: Dense → GELU → Dropout → Dense → num_labels logits."""

    def __init__(self, input_size: int, hidden_size: int, num_labels: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class SafetyGate(nn.Module):
    """Binary safety head: sigmoid score in [0, 1] where 0 = safe, 1 = unsafe.

    Even if the classification head predicts "benign", the safety gate can
    override to "block" if it detects potential danger (defense-in-depth).
    """

    def __init__(self, input_size: int, hidden_size: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.act(self.fc1(x)))
        return torch.sigmoid(self.fc2(out))


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

class GuardrailClassifier(nn.Module):
    """Full guardrail architecture: Encoder → MiddleBlock → Decoder + SafetyGate.

    Architecture layers:
        1. Encoder    – Pre-trained transformer (DistilBERT) + pooling
        2. Middle     – Feature refinement (Dense-LN-GELU ×2 with residual)
        3. Decoder    – Multi-layer classification head  (→ 3-class logits)
        4. SafetyGate – Binary unsafe detector            (→ safety score [0,1])
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int = 3,
        dropout: float = 0.2,
        pooling: str = "cls_mean",
        arch_config: Optional[ArchConfig] = None,
    ) -> None:
        super().__init__()
        if pooling not in POOLING_STRATEGIES:
            raise ValueError(f"pooling must be one of {POOLING_STRATEGIES}, got '{pooling}'")
        self.pooling = pooling
        self.arch_config = arch_config or ArchConfig()

        # Layer 1: Encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        pool_output_size = hidden_size * 2 if pooling == "cls_mean" else hidden_size

        # Layer 2: Middle Block
        self.middle_block = MiddleBlock(
            input_size=pool_output_size,
            intermediate_size=self.arch_config.intermediate_size,
            dropout=dropout,
        )

        # Layer 3: Classification Decoder
        self.classifier = ClassificationDecoder(
            input_size=self.arch_config.intermediate_size,
            hidden_size=self.arch_config.decoder_hidden_size,
            num_labels=num_labels,
            dropout=dropout,
        )

        # Layer 4: Safety Gate
        self.enable_safety_gate = self.arch_config.enable_safety_gate
        if self.enable_safety_gate:
            self.safety_gate = SafetyGate(
                input_size=self.arch_config.intermediate_size,
                hidden_size=self.arch_config.safety_gate_hidden,
                dropout=dropout,
            )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> GuardrailModelOutput:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = pool_encoder_output(outputs.last_hidden_state, attention_mask, self.pooling)
        refined = self.middle_block(pooled)
        logits = self.classifier(refined)

        if self.enable_safety_gate:
            safety_score = self.safety_gate(refined)
        else:
            safety_score = torch.zeros(input_ids.shape[0], 1, device=input_ids.device)

        return GuardrailModelOutput(logits=logits, safety_score=safety_score)


# ---------------------------------------------------------------------------
# Rule-based pre-filter (layered defense)
# ---------------------------------------------------------------------------

_JAILBREAK_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|directives?|rules?|prompts?)",
    r"(disregard|bypass|override|disable)\s+(all\s+)?(safety|guardrail|filter|restriction|rule)",
    r"(you\s+are|act\s+as|pretend\s+(to\s+be|you\'?re))\s+(an?\s+)?(unrestricted|unfiltered|uncensored)",
    r"(new|override|replacement)\s+system\s+(prompt|message|instruction)",
    r"(DAN|jailbreak|developer|debug)\s+mode",
    r"do\s+anything\s+now",
    r"respond\s+without\s+(any\s+)?(restrictions?|limitations?|filters?|safety)",
]
_JAILBREAK_RE = [re.compile(p, re.IGNORECASE) for p in _JAILBREAK_PATTERNS]

_HARMFUL_KEYWORDS = [
    "how to make a bomb", "create malware", "hack into",
    "forge identity", "phishing scam", "ransomware",
    "steal passwords", "bypass security", "exploit vulnerability",
]


@dataclass
class RuleResult:
    triggered: bool = False
    rule_label: Optional[str] = None
    rule_name: Optional[str] = None
    confidence: float = 0.0


def rule_based_prefilter(text: str) -> RuleResult:
    """Lightweight rule layer: catches obvious jailbreak/harmful patterns before model inference."""
    lower = text.lower()
    for i, pattern in enumerate(_JAILBREAK_RE):
        if pattern.search(text):
            return RuleResult(triggered=True, rule_label="jailbreak", rule_name=f"jailbreak_regex_{i}", confidence=0.95)
    for keyword in _HARMFUL_KEYWORDS:
        if keyword in lower:
            return RuleResult(triggered=True, rule_label="harmful", rule_name="harmful_keyword", confidence=0.90)
    return RuleResult()


# ---------------------------------------------------------------------------
# Multi-turn escalation detection
# ---------------------------------------------------------------------------

_MULTI_TURN_ESCALATION_SEQUENCES: List[dict] = [
    {
        "name": "persona_then_exploit",
        "description": "Turn 1 sets up a persona, turn 2 exploits it",
        "priors": [
            re.compile(r"(you\s+are|act\s+as|pretend|imagine|role.?play|let\'?s\s+play)", re.IGNORECASE),
        ],
        "current": re.compile(
            r"(now\s+(as|that|since)|in\s+(that|this)\s+(role|character)|staying\s+in\s+character)"
            r".{0,80}(tell|show|explain|give|write|how|create|generate|make)",
            re.IGNORECASE,
        ),
        "label": "jailbreak",
        "confidence": 0.92,
    },
    {
        "name": "gradual_topic_shift",
        "description": "Benign topic narrowed into dangerous territory step by step",
        "priors": [
            re.compile(r"(tell\s+me\s+about|explain|what\s+is|how\s+does|describe)", re.IGNORECASE),
        ],
        "current": re.compile(
            r"(now\s+specifically|more\s+specifically|going\s+deeper|"
            r"now\s+how\s+(would|could|can|do)\s+(you|we|i|someone))"
            r".{0,80}(harm|attack|exploit|weapon|hack|kill|break\s+into|damage|destroy|poison|bypass)",
            re.IGNORECASE,
        ),
        "label": "harmful",
        "confidence": 0.90,
    },
    {
        "name": "system_prompt_extraction",
        "description": "Probing for system prompt then referring to it",
        "priors": [
            re.compile(
                r"(what\s+are\s+your\s+(instructions?|rules?|guidelines?|system\s+prompt)|"
                r"repeat\s+(your\s+)?(system|initial)\s+(prompt|instructions?|message))",
                re.IGNORECASE,
            ),
        ],
        "current": re.compile(
            r"(ok\s+now\s+ignore|great\s+now\s+ignore|forget\s+(about\s+)?(those|that|them)|"
            r"now\s+(override|disregard|bypass|replace)\s+(those|that|them|it|the))",
            re.IGNORECASE,
        ),
        "label": "jailbreak",
        "confidence": 0.93,
    },
    {
        "name": "hypothetical_then_direct",
        "description": "Hypothetical framing then demanding concrete steps",
        "priors": [
            re.compile(
                r"(hypothetical(ly)?|theoretically|in\s+a\s+(fiction|novel|story|movie|game)|"
                r"for\s+(educational|research|academic)\s+purposes?)",
                re.IGNORECASE,
            ),
        ],
        "current": re.compile(
            r"(now\s+(give|provide|list|write)\s+(me\s+)?(the\s+)?(actual|real|specific|concrete|detailed|step)|"
            r"(ok\s+)?now\s+(for\s+real|seriously|actually|in\s+practice))",
            re.IGNORECASE,
        ),
        "label": "jailbreak",
        "confidence": 0.91,
    },
]


def multi_turn_rule_check(current_text: str, prior_texts: List[str]) -> RuleResult:
    """Check whether the current prompt + prior conversation history
    matches any known multi-turn attack escalation sequence."""
    if not prior_texts:
        return RuleResult()

    for seq in _MULTI_TURN_ESCALATION_SEQUENCES:
        if not seq["current"].search(current_text):
            continue
        for prior in prior_texts:
            for prior_re in seq["priors"]:
                if prior_re.search(prior):
                    return RuleResult(
                        triggered=True,
                        rule_label=seq["label"],
                        rule_name=f"multi_turn_{seq['name']}",
                        confidence=seq["confidence"],
                    )
    return RuleResult()
