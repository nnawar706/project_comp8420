"""
Text preprocessing + PII redaction.

Two cleaning paths, on purpose:
  * clean_for_classical()  -> lowercased, normalised (for TF-IDF / classical models)
  * clean_for_llm()        -> light touch, preserves case (for NER and the LLM)

PII redaction is a cheap, ethically-important step . It returns both the redacted string and the spans it found, so the
same regex doubles as silver labels for NER evaluation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Regex patterns for PII / domain entities
# --------------------------------------------------------------------------- #
PATTERNS: dict[str, re.Pattern] = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE": re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?){2,4}\d{2,4}(?!\d)"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "ORDER_NUMBER": re.compile(r"\b(?:order|ord|#)\s*[:#]?\s*([A-Z]{0,3}\d{3,12})\b", re.I),
    "INVOICE_NUMBER": re.compile(r"\b(?:invoice|inv)\s*[:#]?\s*([A-Z]{0,3}\d{3,12})\b", re.I),
    "TRACKING_NUMBER": re.compile(r"\b(?:tracking|track)\s*[:#]?\s*([A-Z0-9]{8,30})\b", re.I),
}

# Order applied to redaction (more specific first to avoid e.g. an order number
# being swallowed by the generic phone pattern).
_REDACT_ORDER = ["EMAIL", "CREDIT_CARD", "ORDER_NUMBER", "INVOICE_NUMBER", "TRACKING_NUMBER", "PHONE"]

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MULTISPACE_RE = re.compile(r"\s+")
_PLACEHOLDER_RE = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")  # Bitext {{Order Number}} style


@dataclass
class PiiSpan:
    label: str
    value: str
    start: int
    end: int


def find_pii(text: str) -> list[PiiSpan]:
    """Locate PII / domain entity spans (used for redaction and as NER silver labels)."""
    spans: list[PiiSpan] = []
    for label in _REDACT_ORDER:
        for m in PATTERNS[label].finditer(text):
            spans.append(PiiSpan(label, m.group(0), m.start(), m.end()))
    # Resolve overlaps: keep the earliest, longest match.
    spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
    chosen: list[PiiSpan] = []
    occupied: list[tuple[int, int]] = []
    for s in spans:
        if any(not (s.end <= a or s.start >= b) for a, b in occupied):
            continue
        chosen.append(s)
        occupied.append((s.start, s.end))
    return chosen


def redact_pii(text: str) -> tuple[str, list[PiiSpan]]:
    """Replace PII with <LABEL> placeholders. Returns (redacted_text, spans)."""
    spans = find_pii(text)
    redacted = text
    for s in sorted(spans, key=lambda x: x.start, reverse=True):
        redacted = redacted[: s.start] + f"<{s.label}>" + redacted[s.end :]
    return redacted, spans


# --------------------------------------------------------------------------- #
# Bitext placeholder handling
# --------------------------------------------------------------------------- #
def extract_placeholder_entities(text: str) -> list[dict]:
    """
    Bitext messages contain gold spans like '{{Order Number}}'.
    We treat these as free silver labels for NER evaluation.
    Returns spans measured against the text with braces stripped.
    """
    out, clean, cursor = [], [], 0
    for m in _PLACEHOLDER_RE.finditer(text):
        clean.append(text[cursor:m.start()])
        label = m.group(1).strip().upper().replace(" ", "_")
        start = sum(len(c) for c in clean)
        value = m.group(1).strip()
        clean.append(value)
        out.append({"text": value, "label": label, "start": start, "end": start + len(value)})
        cursor = m.end()
    clean.append(text[cursor:])
    return out


_FAKE_VALUES = {
    "ORDER_NUMBER": "48213", "ORDER NUMBER": "48213",
    "INVOICE_NUMBER": "INV-90871", "INVOICE NUMBER": "INV-90871",
    "PERSON_NAME": "Alex Morgan", "PERSON NAME": "Alex Morgan",
    "DELIVERY_ADDRESS": "12 Baker Street, London",
    "PRODUCT": "wireless headphones", "DATE": "March 3rd",
}


def fill_placeholders(text: str, mapping: dict | None = None) -> str:
    """Substitute Bitext {{placeholders}} with realistic values for demos."""
    mapping = mapping or {}

    def _sub(m: re.Match) -> str:
        key = m.group(1).strip().upper().replace(" ", "_")
        return str(mapping.get(key, _FAKE_VALUES.get(key, _FAKE_VALUES.get(m.group(1).strip().upper(), m.group(1).strip()))))

    return _PLACEHOLDER_RE.sub(_sub, text)


def strip_placeholders(text: str) -> str:
    """Remove the {{ }} braces, keeping the inner text."""
    return _PLACEHOLDER_RE.sub(lambda m: m.group(1).strip(), text)


# --------------------------------------------------------------------------- #
# Cleaning paths
# --------------------------------------------------------------------------- #
def clean_for_llm(text: str) -> str:
    """Light clean: drop URLs, collapse whitespace, keep case + punctuation."""
    text = strip_placeholders(str(text))
    text = _URL_RE.sub(" ", text)
    return _MULTISPACE_RE.sub(" ", text).strip()


def clean_for_classical(text: str) -> str:
    """Aggressive clean for TF-IDF: lowercase, strip URLs/punctuation noise."""
    text = clean_for_llm(text).lower()
    text = re.sub(r"[^a-z0-9\s'#]", " ", text)
    return _MULTISPACE_RE.sub(" ", text).strip()


if __name__ == "__main__":
    sample = "My order #48213 hasn't arrived. Email me at jo@acme.io or call 0412 345 678."
    red, spans = redact_pii(sample)
    print("Redacted:", red)
    for sp in spans:
        print("  ", sp.label, "->", sp.value)
    print("Classical:", clean_for_classical(sample))
