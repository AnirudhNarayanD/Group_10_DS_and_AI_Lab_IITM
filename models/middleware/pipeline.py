"""Hybrid Guardrail Pipeline: layered defense with PII detection,
rule-based pre-filtering, multi-turn context tracking, model inference,
and output guardrail scanning.

This module defines how the guardrail classifier integrates into an LLM inference
pipeline.  The design follows a multi-layer defense strategy with conversation-
aware multi-turn prompt injection detection:

Layer -1:  PII Detection            (regex-based PII scan: redact/mask/hash/block)
Layer 0:   Rule-based pre-filter     (fast regex/keyword checks)
Layer 0.5: Multi-turn rule patterns  (cross-turn escalation sequences)
Layer 1:   Encoder                   (DeBERTa / DistilBERT + pooling)
Layer 2:   Middle Block              (Dense-LN-GELU feature refinement)
Layer 3:   Classification Decoder    (multi-layer head → 3-class logits)
Layer 4:   Safety Gate               (binary unsafe detector → override)
Layer 5:   Threshold Decision        (configurable allow/block policy)
Layer 6:   Escalation Risk           (cumulative cross-turn risk accumulator)
Layer 7:   Output Guardrail          (run classifier on LLM response to catch harmful output)

Multi-turn detection mechanisms:
  1. Cross-turn rule patterns   — regex sequences that match across turns
  2. Context-aware model input  — [prev] [SEP] [current] fed to encoder
  3. Escalation risk scoring    — decayed cumulative attack probability

                ┌─────────────────┐
                │  User Prompt    │
                └───────┬─────────┘
                        │
              ┌─────────▼──────────┐
              │ Layer -1: PII      │
              │ Detection          │──── Block (SSN/API key) ──► BLOCKED
              │ (redact/mask/hash) │     or sanitize & continue
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────┐
              │ ConversationContext │
              │ (sliding window of │
              │  prior N turns)    │
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────┐
              │ Layer 0: Rule      │
              │ Pre-Filter         │──── Block (obvious) ──► BLOCKED
              └─────────┬──────────┘
                        │ Pass
              ┌─────────▼──────────┐
              │ Layer 0.5: Multi-  │
              │ Turn Rule Patterns │──── Block (escalation  ──► BLOCKED
              │ (cross-turn regex) │     sequence detected)
              └─────────┬──────────┘
                        │ Pass
              ┌─────────▼──────────┐
              │ Context-Aware      │
              │ Model Input:       │
              │ [prev₁][SEP]       │
              │ [prev₂][SEP]       │
              │ [current]          │
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────┐
              │ Layers 1-4: Model  │
              │ (Encoder→Middle→   │
              │  Decoder+SafetyGate│
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────┐
              │ Layer 5: Threshold │
              │ + Safety Gate      │──── Block ──► BLOCKED
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────┐
              │ Layer 6: Escalation│
              │ Risk Accumulator   │──── Block (rising ──► BLOCKED
              │ (cross-turn decay) │     cumulative risk)
              └─────────┬──────────┘
                        │ Allow
              ┌─────────▼──────────┐
              │ LLM Inference      │
              └─────────┬──────────┘
                        │
              ┌─────────▼──────────┐
              │ Layer 7: Output    │
              │ Guardrail (reuse   │──── Block (harmful ──► SANITIZED
              │ classifier on LLM  │     LLM output)
              │ response + PII)    │
              └─────────┬──────────┘
                        │ Safe
              ┌─────────▼──────────┐
              │  Return Response   │
              └────────────────────┘
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

from models.common import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    ThresholdConfig,
    build_tokenizer,
    choose_device,
    normalize_text_for_model,
)
from models.middleware.classifier import (
    ArchConfig,
    GuardrailClassifier,
    RuleResult,
    multi_turn_rule_check,
    rule_based_prefilter,
)
from models.middleware.pii import PIIConfig, PIIScanResult, PIIStrategy, scan_pii


@dataclass
class GuardrailDecision:
    """Result of the guardrail pipeline for a single prompt."""
    action: str  # "allow" or "block"
    label: str   # "benign", "jailbreak", "harmful"
    confidence: float
    layer_triggered: str  # "rule_filter", "model_classifier", "safety_gate",
                          # "multi_turn_rule", "escalation_risk", or "none"
    rule_name: Optional[str] = None
    model_probabilities: Optional[Dict[str, float]] = None
    safety_gate_score: Optional[float] = None
    escalation_risk: Optional[float] = None
    pii_detections: Optional[List[dict]] = None
    pii_sanitized_text: Optional[str] = None
    turn_number: Optional[int] = None
    latency_ms: float = 0.0
    threshold_strategy: str = "argmax"


# ---------------------------------------------------------------------------
# Multi-turn conversation tracking
# ---------------------------------------------------------------------------

@dataclass
class MultiTurnConfig:
    """Configuration for multi-turn prompt injection detection."""
    enabled: bool = True
    context_window: int = 5
    enable_context_model: bool = True
    enable_escalation_risk: bool = True
    escalation_threshold: float = 0.6
    escalation_decay: float = 0.85
    context_separator: str = " [SEP] "


@dataclass
class _TurnRecord:
    """Internal record of a single conversation turn."""
    prompt_text: str
    decision: GuardrailDecision
    attack_prob: float


class ConversationContext:
    """Sliding-window conversation history tracker.

    Maintains the last ``max_turns`` prompts along with their guardrail
    decisions.  Used by the pipeline to:

    1. Feed conversation context to the transformer model so it can detect
       multi-staged prompt injection (context-aware model input).
    2. Run multi-turn rule patterns across the sequence of prompts.
    3. Compute an escalation risk score that accumulates across turns.
    """

    def __init__(self, max_turns: int = 5) -> None:
        self.max_turns = max_turns
        self._history: List[_TurnRecord] = []

    def add_turn(self, prompt: str, decision: GuardrailDecision, attack_prob: float) -> None:
        self._history.append(_TurnRecord(prompt, decision, attack_prob))
        if len(self._history) > self.max_turns:
            self._history = self._history[-self.max_turns:]

    @property
    def turn_count(self) -> int:
        return len(self._history)

    @property
    def prior_prompts(self) -> List[str]:
        return [t.prompt_text for t in self._history]

    def build_context_input(self, current_prompt: str, separator: str = " [SEP] ") -> str:
        parts = self.prior_prompts + [current_prompt]
        return separator.join(parts)

    def compute_escalation_risk(self, current_attack_prob: float, decay: float = 0.85) -> float:
        probs = [t.attack_prob for t in self._history] + [current_attack_prob]
        n = len(probs)
        if n == 0:
            return 0.0
        weights = [decay ** (n - 1 - i) for i in range(n)]
        weighted_sum = sum(p * w for p, w in zip(probs, weights))
        total_weight = sum(weights)
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def reset(self) -> None:
        self._history.clear()


@dataclass
class PipelineConfig:
    """Configuration for the guardrail pipeline."""
    checkpoint_path: Path
    threshold_config: ThresholdConfig = field(default_factory=ThresholdConfig)
    enable_rule_filter: bool = True
    max_length: int = 512
    safety_gate_threshold: float = 0.5
    multi_turn: MultiTurnConfig = field(default_factory=MultiTurnConfig)
    pii_config: Optional[PIIConfig] = None
    log_decisions: bool = False


class GuardrailPipeline:
    """Production-ready guardrail pipeline combining rules + ML classifier
    with multi-turn conversation context tracking.

    Supports both single-turn and multi-turn classification:
        - ``classify(prompt)``  — single-turn (stateless)
        - ``classify_turn(prompt)``  — multi-turn (context-aware, stateful)
        - ``classify_turn(prompt, session_id)``  — multi-session support
    """

    def __init__(
        self,
        model: GuardrailClassifier,
        tokenizer,
        device: torch.device,
        config: PipelineConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.decision_log: List[dict] = []
        self._contexts: Dict[str, ConversationContext] = {}
        self._default_context = ConversationContext(
            max_turns=config.multi_turn.context_window,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        threshold_config: Optional[ThresholdConfig] = None,
        enable_rule_filter: bool = True,
        safety_gate_threshold: float = 0.5,
        multi_turn_config: Optional[MultiTurnConfig] = None,
        pii_config: Optional[PIIConfig] = None,
    ) -> "GuardrailPipeline":
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        model_name = checkpoint["model_name"]
        max_length = int(checkpoint.get("max_length", 512))
        pooling = checkpoint.get("pooling", "cls_mean")

        arch_config_dict = checkpoint.get("arch_config", None)
        arch_cfg = ArchConfig(**arch_config_dict) if arch_config_dict else ArchConfig()

        device = choose_device()
        model = GuardrailClassifier(
            model_name=model_name,
            num_labels=len(LABEL_TO_ID),
            pooling=pooling,
            arch_config=arch_cfg,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        tokenizer = build_tokenizer(model_name)

        config = PipelineConfig(
            checkpoint_path=checkpoint_path,
            threshold_config=threshold_config or ThresholdConfig(),
            enable_rule_filter=enable_rule_filter,
            max_length=max_length,
            safety_gate_threshold=safety_gate_threshold,
            multi_turn=multi_turn_config or MultiTurnConfig(),
            pii_config=pii_config,
        )

        return cls(model=model, tokenizer=tokenizer, device=device, config=config)

    def _run_pii_scan(self, text: str) -> Optional[PIIScanResult]:
        """Run PII detection if configured."""
        if self.config.pii_config is None:
            return None
        return scan_pii(text, self.config.pii_config)

    def classify(self, prompt: str) -> GuardrailDecision:
        """Run the full layered guardrail classification on a single prompt."""
        start = time.perf_counter()

        # Layer -1: PII detection
        pii_result = self._run_pii_scan(prompt)
        if pii_result is not None and pii_result.has_pii:
            pii_dicts = [
                {"type": d.pii_type, "strategy": d.strategy.value, "replacement": d.replacement}
                for d in pii_result.detections
            ]
            if pii_result.should_block:
                return GuardrailDecision(
                    action="block",
                    label="harmful",
                    confidence=1.0,
                    layer_triggered="pii_filter",
                    pii_detections=pii_dicts,
                    latency_ms=round((time.perf_counter() - start) * 1000, 2),
                    threshold_strategy=self.config.threshold_config.strategy,
                )
            # Non-blocking PII: sanitize text and continue classification
            prompt = pii_result.sanitized_text
        else:
            pii_dicts = None

        # Layer 0: Rule-based pre-filter
        if self.config.enable_rule_filter:
            rule_result = rule_based_prefilter(prompt)
            if rule_result.triggered:
                decision = GuardrailDecision(
                    action="block",
                    label=rule_result.rule_label or "jailbreak",
                    confidence=rule_result.confidence,
                    layer_triggered="rule_filter",
                    rule_name=rule_result.rule_name,
                    latency_ms=round((time.perf_counter() - start) * 1000, 2),
                    threshold_strategy=self.config.threshold_config.strategy,
                )
                if self.config.log_decisions:
                    self.decision_log.append(asdict(decision))
                return decision

        # Layers 1-4: Model inference
        normalized = normalize_text_for_model(prompt)
        enc = self.tokenizer(
            normalized,
            truncation=True,
            max_length=self.config.max_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(output.logits, dim=-1)
            safety_score = float(output.safety_score[0, 0].item())

        # Layer 5: Threshold decision + safety gate override
        pred_idx = self.config.threshold_config.apply(probs)
        pred_label = ID_TO_LABEL[int(pred_idx.item())]
        confidence = float(probs[0, int(pred_idx.item())].item())

        model_probs = {
            ID_TO_LABEL[j]: round(float(probs[0, j].item()), 6)
            for j in range(len(ID_TO_LABEL))
        }

        action = "allow" if pred_label == "benign" else "block"
        layer_triggered = "model_classifier"

        if action == "allow" and safety_score >= self.config.safety_gate_threshold:
            action = "block"
            jailbreak_prob = probs[0, LABEL_TO_ID["jailbreak"]].item()
            harmful_prob = probs[0, LABEL_TO_ID["harmful"]].item()
            pred_label = "jailbreak" if jailbreak_prob >= harmful_prob else "harmful"
            confidence = round(safety_score, 4)
            layer_triggered = "safety_gate"

        decision = GuardrailDecision(
            action=action,
            label=pred_label,
            confidence=round(confidence, 4),
            layer_triggered=layer_triggered,
            model_probabilities=model_probs,
            safety_gate_score=round(safety_score, 4),
            pii_detections=pii_dicts,
            pii_sanitized_text=pii_result.sanitized_text if pii_result and pii_result.has_pii else None,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            threshold_strategy=self.config.threshold_config.strategy,
        )

        if self.config.log_decisions:
            self.decision_log.append(asdict(decision))
        return decision

    def classify_batch(self, prompts: List[str]) -> List[GuardrailDecision]:
        return [self.classify(prompt) for prompt in prompts]

    # ------------------------------------------------------------------
    # Multi-turn context-aware classification
    # ------------------------------------------------------------------

    def _get_context(self, session_id: Optional[str] = None) -> ConversationContext:
        if session_id is None:
            return self._default_context
        if session_id not in self._contexts:
            self._contexts[session_id] = ConversationContext(
                max_turns=self.config.multi_turn.context_window,
            )
        return self._contexts[session_id]

    def classify_turn(
        self,
        prompt: str,
        session_id: Optional[str] = None,
    ) -> GuardrailDecision:
        """Context-aware classification that tracks multi-turn conversation."""
        mt = self.config.multi_turn
        ctx = self._get_context(session_id)
        start = time.perf_counter()

        # Layer -1: PII detection
        pii_result = self._run_pii_scan(prompt)
        if pii_result is not None and pii_result.has_pii:
            pii_dicts = [
                {"type": d.pii_type, "strategy": d.strategy.value, "replacement": d.replacement}
                for d in pii_result.detections
            ]
            if pii_result.should_block:
                decision = GuardrailDecision(
                    action="block",
                    label="harmful",
                    confidence=1.0,
                    layer_triggered="pii_filter",
                    pii_detections=pii_dicts,
                    turn_number=ctx.turn_count + 1,
                    latency_ms=round((time.perf_counter() - start) * 1000, 2),
                    threshold_strategy=self.config.threshold_config.strategy,
                )
                ctx.add_turn(prompt, decision, attack_prob=0.0)
                if self.config.log_decisions:
                    self.decision_log.append(asdict(decision))
                return decision
            prompt = pii_result.sanitized_text
        else:
            pii_dicts = None

        # Layer 0: Single-turn rule pre-filter
        if self.config.enable_rule_filter:
            rule_result = rule_based_prefilter(prompt)
            if rule_result.triggered:
                decision = GuardrailDecision(
                    action="block",
                    label=rule_result.rule_label or "jailbreak",
                    confidence=rule_result.confidence,
                    layer_triggered="rule_filter",
                    rule_name=rule_result.rule_name,
                    turn_number=ctx.turn_count + 1,
                    latency_ms=round((time.perf_counter() - start) * 1000, 2),
                    threshold_strategy=self.config.threshold_config.strategy,
                )
                ctx.add_turn(prompt, decision, attack_prob=rule_result.confidence)
                if self.config.log_decisions:
                    self.decision_log.append(asdict(decision))
                return decision

        # Layer 0.5: Multi-turn rule escalation check
        if mt.enabled and ctx.turn_count > 0:
            mt_result = multi_turn_rule_check(prompt, ctx.prior_prompts)
            if mt_result.triggered:
                decision = GuardrailDecision(
                    action="block",
                    label=mt_result.rule_label or "jailbreak",
                    confidence=mt_result.confidence,
                    layer_triggered="multi_turn_rule",
                    rule_name=mt_result.rule_name,
                    turn_number=ctx.turn_count + 1,
                    latency_ms=round((time.perf_counter() - start) * 1000, 2),
                    threshold_strategy=self.config.threshold_config.strategy,
                )
                ctx.add_turn(prompt, decision, attack_prob=mt_result.confidence)
                if self.config.log_decisions:
                    self.decision_log.append(asdict(decision))
                return decision

        # Layers 1-4: Model inference (context-aware)
        if mt.enabled and mt.enable_context_model and ctx.turn_count > 0:
            model_input = ctx.build_context_input(prompt, separator=mt.context_separator)
        else:
            model_input = prompt

        normalized = normalize_text_for_model(model_input)
        enc = self.tokenizer(
            normalized,
            truncation=True,
            max_length=self.config.max_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(output.logits, dim=-1)
            safety_score = float(output.safety_score[0, 0].item())

        # Layer 5: Threshold decision + safety gate
        pred_idx = self.config.threshold_config.apply(probs)
        pred_label = ID_TO_LABEL[int(pred_idx.item())]
        confidence = float(probs[0, int(pred_idx.item())].item())

        model_probs = {
            ID_TO_LABEL[j]: round(float(probs[0, j].item()), 6)
            for j in range(len(ID_TO_LABEL))
        }

        action = "allow" if pred_label == "benign" else "block"
        layer_triggered = "model_classifier"

        if action == "allow" and safety_score >= self.config.safety_gate_threshold:
            action = "block"
            jailbreak_prob = probs[0, LABEL_TO_ID["jailbreak"]].item()
            harmful_prob = probs[0, LABEL_TO_ID["harmful"]].item()
            pred_label = "jailbreak" if jailbreak_prob >= harmful_prob else "harmful"
            confidence = round(safety_score, 4)
            layer_triggered = "safety_gate"

        # Layer 6: Multi-turn escalation risk check
        current_attack_prob = max(
            probs[0, LABEL_TO_ID["jailbreak"]].item(),
            probs[0, LABEL_TO_ID["harmful"]].item(),
        )
        escalation_risk = ctx.compute_escalation_risk(
            current_attack_prob, decay=mt.escalation_decay,
        )

        if (
            mt.enabled
            and mt.enable_escalation_risk
            and action == "allow"
            and escalation_risk >= mt.escalation_threshold
            and ctx.turn_count >= 2
        ):
            action = "block"
            jailbreak_prob = probs[0, LABEL_TO_ID["jailbreak"]].item()
            harmful_prob = probs[0, LABEL_TO_ID["harmful"]].item()
            pred_label = "jailbreak" if jailbreak_prob >= harmful_prob else "harmful"
            confidence = round(escalation_risk, 4)
            layer_triggered = "escalation_risk"

        decision = GuardrailDecision(
            action=action,
            label=pred_label,
            confidence=round(confidence, 4),
            layer_triggered=layer_triggered,
            model_probabilities=model_probs,
            safety_gate_score=round(safety_score, 4),
            escalation_risk=round(escalation_risk, 4),
            pii_detections=pii_dicts,
            pii_sanitized_text=pii_result.sanitized_text if pii_result and pii_result.has_pii else None,
            turn_number=ctx.turn_count + 1,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            threshold_strategy=self.config.threshold_config.strategy,
        )

        ctx.add_turn(prompt, decision, attack_prob=current_attack_prob)

        if self.config.log_decisions:
            self.decision_log.append(asdict(decision))
        return decision

    def reset_session(self, session_id: Optional[str] = None) -> None:
        if session_id is None:
            self._default_context.reset()
        elif session_id in self._contexts:
            self._contexts[session_id].reset()

    def reset_all_sessions(self) -> None:
        self._default_context.reset()
        self._contexts.clear()

    def get_decision_summary(self) -> dict:
        if not self.decision_log:
            return {"total": 0}
        total = len(self.decision_log)
        blocked = sum(1 for d in self.decision_log if d["action"] == "block")
        by_layer = {}
        for d in self.decision_log:
            layer = d["layer_triggered"]
            by_layer[layer] = by_layer.get(layer, 0) + 1
        avg_latency = sum(d["latency_ms"] for d in self.decision_log) / total
        return {
            "total": total,
            "allowed": total - blocked,
            "blocked": blocked,
            "block_rate": round(blocked / total, 4),
            "by_layer": by_layer,
            "avg_latency_ms": round(avg_latency, 2),
        }

    def save_decision_log(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.decision_log, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Layer 7: Output Guardrail
    # ------------------------------------------------------------------

    def classify_output(self, response_text: str) -> GuardrailDecision:
        """Run the guardrail classifier on an LLM response (Layer 7).

        This catches harmful or jailbreak content that slipped through the
        input guardrails and was generated by the LLM.  It also runs PII
        detection on the response to prevent PII leakage in model output.
        """
        start = time.perf_counter()

        # PII scan on output (prevent PII leakage)
        pii_result = self._run_pii_scan(response_text)
        if pii_result is not None and pii_result.has_pii:
            pii_dicts = [
                {"type": d.pii_type, "strategy": d.strategy.value, "replacement": d.replacement}
                for d in pii_result.detections
            ]
            if pii_result.should_block:
                return GuardrailDecision(
                    action="block",
                    label="harmful",
                    confidence=1.0,
                    layer_triggered="output_guardrail_pii",
                    pii_detections=pii_dicts,
                    latency_ms=round((time.perf_counter() - start) * 1000, 2),
                    threshold_strategy=self.config.threshold_config.strategy,
                )
            response_text = pii_result.sanitized_text
        else:
            pii_dicts = None

        # Classify the LLM response text for safety
        normalized = normalize_text_for_model(response_text)
        enc = self.tokenizer(
            normalized,
            truncation=True,
            max_length=self.config.max_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(output.logits, dim=-1)
            safety_score = float(output.safety_score[0, 0].item())

        pred_idx = self.config.threshold_config.apply(probs)
        pred_label = ID_TO_LABEL[int(pred_idx.item())]
        confidence = float(probs[0, int(pred_idx.item())].item())

        model_probs = {
            ID_TO_LABEL[j]: round(float(probs[0, j].item()), 6)
            for j in range(len(ID_TO_LABEL))
        }

        action = "allow" if pred_label == "benign" else "block"
        layer_triggered = "output_guardrail"

        if action == "allow" and safety_score >= self.config.safety_gate_threshold:
            action = "block"
            jailbreak_prob = probs[0, LABEL_TO_ID["jailbreak"]].item()
            harmful_prob = probs[0, LABEL_TO_ID["harmful"]].item()
            pred_label = "jailbreak" if jailbreak_prob >= harmful_prob else "harmful"
            confidence = round(safety_score, 4)
            layer_triggered = "output_guardrail_gate"

        decision = GuardrailDecision(
            action=action,
            label=pred_label,
            confidence=round(confidence, 4),
            layer_triggered=layer_triggered,
            model_probabilities=model_probs,
            safety_gate_score=round(safety_score, 4),
            pii_detections=pii_dicts,
            pii_sanitized_text=pii_result.sanitized_text if pii_result and pii_result.has_pii else None,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            threshold_strategy=self.config.threshold_config.strategy,
        )

        if self.config.log_decisions:
            self.decision_log.append(asdict(decision))
        return decision


def demonstrate_llm_integration():
    """Pseudocode demonstrating how the guardrail fits into an LLM serving pipeline.

    ```python
    from models.middleware.pipeline import (
        GuardrailPipeline, MultiTurnConfig,
    )
    from models.middleware.pii import PIIConfig, PIIStrategy
    from models.common import ThresholdConfig

    # Initialize once at server startup
    guardrail = GuardrailPipeline.from_checkpoint(
        "model/best_model.pt",
        threshold_config=ThresholdConfig(
            strategy="safety_biased",
            block_threshold=0.4,
        ),
        enable_rule_filter=True,
        multi_turn_config=MultiTurnConfig(
            enabled=True,
            context_window=5,
            escalation_threshold=0.6,
        ),
    )
    # Enable PII detection (redact by default, block SSN/API keys)
    guardrail.config.pii_config = PIIConfig(
        rules={
            "*": PIIStrategy.REDACT,
            "ssn": PIIStrategy.BLOCK,
            "api_key": PIIStrategy.BLOCK,
            "credit_card": PIIStrategy.MASK,
        },
    )

    @app.post("/chat")
    async def chat(request: ChatRequest):
        # Layer -1 → Layer 6: Input guardrail (PII + rules + model + escalation)
        decision = guardrail.classify_turn(
            request.prompt,
            session_id=request.session_id,
        )
        if decision.action == "block":
            return ChatResponse(
                text="I'm unable to assist with that request.",
                guardrail_decision=decision,
            )

        # Use sanitized text (PII redacted) if available
        safe_prompt = decision.pii_sanitized_text or request.prompt
        response = await llm.generate(safe_prompt)

        # Layer 7: Output guardrail (classifier + PII scan on LLM response)
        output_decision = guardrail.classify_output(response.text)
        if output_decision.action == "block":
            return ChatResponse(
                text="I'm unable to provide that information.",
                guardrail_decision=output_decision,
            )

        # Return the sanitized output (PII redacted) if present
        safe_response = output_decision.pii_sanitized_text or response.text
        return ChatResponse(text=safe_response, guardrail_decision=output_decision)
    ```
    """
    pass
