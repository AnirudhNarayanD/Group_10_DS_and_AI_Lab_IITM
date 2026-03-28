"""PII Detection Middleware — deterministic regex-based PII scanning.

Detects and handles Personally Identifiable Information in prompts and
LLM responses.  Supports four strategies:
  - redact:  Replace with [REDACTED_{PII_TYPE}]
  - mask:    Partially obscure (e.g. last 4 digits visible)
  - hash:    Replace with SHA-256 deterministic hash
  - block:   Flag for blocking (caller decides action)

Inspired by LangChain's PIIMiddleware pattern.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class PIIStrategy(Enum):
    REDACT = "redact"
    MASK = "mask"
    HASH = "hash"
    BLOCK = "block"


@dataclass
class PIIDetection:
    """A single PII match found in text."""
    pii_type: str
    matched_text: str
    start: int
    end: int
    strategy: PIIStrategy
    replacement: Optional[str] = None


@dataclass
class PIIScanResult:
    """Result of scanning text for PII."""
    has_pii: bool = False
    detections: List[PIIDetection] = field(default_factory=list)
    sanitized_text: Optional[str] = None
    should_block: bool = False
    summary: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PII patterns — compiled regexes for common PII types
# ---------------------------------------------------------------------------

_PII_PATTERNS: Dict[str, re.Pattern] = {
    "email": re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    ),
    "phone_us": re.compile(
        r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"
    ),
    "ssn": re.compile(
        r"\b\d{3}-\d{2}-\d{4}\b"
    ),
    "credit_card": re.compile(
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
    ),
    "ip_address": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ),
    "api_key": re.compile(
        r"\b(?:sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{36}|AKIA[A-Z0-9]{16})\b"
    ),
}


def _redact(pii_type: str, _matched: str) -> str:
    return f"[REDACTED_{pii_type.upper()}]"


def _mask(pii_type: str, matched: str) -> str:
    if pii_type == "credit_card":
        digits = re.sub(r"\D", "", matched)
        return "*" * (len(digits) - 4) + digits[-4:]
    if pii_type == "phone_us":
        digits = re.sub(r"\D", "", matched)
        if len(digits) >= 4:
            return "*" * (len(digits) - 4) + digits[-4:]
    if pii_type == "ssn":
        return "***-**-" + matched[-4:]
    if pii_type == "email":
        local, domain = matched.rsplit("@", 1)
        visible = local[0] if local else ""
        return f"{visible}***@{domain}"
    return "*" * max(len(matched) - 4, 0) + matched[-4:]


def _hash_pii(_pii_type: str, matched: str) -> str:
    digest = hashlib.sha256(matched.encode("utf-8")).hexdigest()[:16]
    return f"[HASH:{digest}]"


_STRATEGY_FN = {
    PIIStrategy.REDACT: _redact,
    PIIStrategy.MASK: _mask,
    PIIStrategy.HASH: _hash_pii,
    PIIStrategy.BLOCK: _redact,  # block still redacts in sanitized copy
}


# ---------------------------------------------------------------------------
# PII Scanner
# ---------------------------------------------------------------------------

@dataclass
class PIIConfig:
    """Configuration for PII detection.

    ``rules`` maps PII type names to strategies.  Any PII type not listed
    is ignored.  Use ``"*"`` as key to set a default strategy.
    """
    rules: Dict[str, PIIStrategy] = field(default_factory=lambda: {
        "email": PIIStrategy.REDACT,
        "phone_us": PIIStrategy.REDACT,
        "ssn": PIIStrategy.BLOCK,
        "credit_card": PIIStrategy.MASK,
        "ip_address": PIIStrategy.REDACT,
        "api_key": PIIStrategy.BLOCK,
    })
    custom_patterns: Dict[str, re.Pattern] = field(default_factory=dict)


def scan_pii(text: str, config: Optional[PIIConfig] = None) -> PIIScanResult:
    """Scan ``text`` for PII and return detections + sanitized text."""
    cfg = config or PIIConfig()
    detections: List[PIIDetection] = []

    all_patterns = dict(_PII_PATTERNS)
    all_patterns.update(cfg.custom_patterns)

    default_strategy = cfg.rules.get("*")

    for pii_type, pattern in all_patterns.items():
        strategy = cfg.rules.get(pii_type, default_strategy)
        if strategy is None:
            continue
        for m in pattern.finditer(text):
            matched = m.group()
            replacement_fn = _STRATEGY_FN[strategy]
            replacement = replacement_fn(pii_type, matched)
            detections.append(PIIDetection(
                pii_type=pii_type,
                matched_text=matched,
                start=m.start(),
                end=m.end(),
                strategy=strategy,
                replacement=replacement,
            ))

    if not detections:
        return PIIScanResult(has_pii=False, sanitized_text=text)

    # Sort by position (reverse) for replacement without offset issues
    detections.sort(key=lambda d: d.start, reverse=True)
    sanitized = text
    for d in detections:
        sanitized = sanitized[:d.start] + (d.replacement or "") + sanitized[d.end:]

    # Re-sort chronologically for output
    detections.sort(key=lambda d: d.start)

    should_block = any(d.strategy == PIIStrategy.BLOCK for d in detections)
    summary: Dict[str, int] = {}
    for d in detections:
        summary[d.pii_type] = summary.get(d.pii_type, 0) + 1

    return PIIScanResult(
        has_pii=True,
        detections=detections,
        sanitized_text=sanitized,
        should_block=should_block,
        summary=summary,
    )
