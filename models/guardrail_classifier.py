"""Backward-compatible re-export shim.

All guardrail-specific code now lives in models/middleware/.
This file re-exports symbols so existing imports continue to work.
"""
# Shared utilities (common to both baseline and middleware)
from models.common import (  # noqa: F401
    ID_TO_LABEL,
    LABEL_TO_ID,
    POOLING_STRATEGIES,
    GuardrailModelOutput,
    ThresholdConfig,
    TokenizationConfig,
    build_tokenizer,
    choose_device,
    compute_truncation_stats,
    load_json_records,
    normalize_text_for_model,
    set_seed,
    validate_records,
)

# Middleware-specific (guardrail architecture + rules)
from models.middleware.classifier import (  # noqa: F401
    ArchConfig,
    ClassificationDecoder,
    GuardrailClassifier,
    MiddleBlock,
    RuleResult,
    SafetyGate,
    multi_turn_rule_check,
    rule_based_prefilter,
)
