from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass
class AnonymizationRule:
    label: str
    pattern: str


DEFAULT_RULES: list[AnonymizationRule] = [
    AnonymizationRule("CPF", r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"),
    AnonymizationRule("CPF", r"\b\d{11}\b"),
    AnonymizationRule("CNPJ", r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b"),
    AnonymizationRule("EMAIL", r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    AnonymizationRule("PHONE", r"\b\+?\d{1,3}\s?\(?\d{2}\)?\s?\d{4,5}-\d{4}\b"),
    AnonymizationRule("DATE", r"\b\d{2}/\d{2}/\d{4}\b"),
    AnonymizationRule("DATE", r"\b\d{4}-\d{2}-\d{2}\b"),
    AnonymizationRule("RG", r"\b\d{1,2}\.\d{3}\.\d{3}-?\d{1}\b"),
    AnonymizationRule("RG", r"\b\d{7,9}\b"),
    AnonymizationRule("CNS", r"\b\d{3}\s?\d{4}\s?\d{4}\s?\d{4}\b"),
    AnonymizationRule("CEP", r"\b\d{5}-\d{3}\b"),
]


def apply_blocklist(text: str, names: Sequence[str]) -> str:
    cleaned = text
    for name in names:
        if not name.strip():
            continue
        pattern = re.compile(rf"\b{re.escape(name.strip())}\b", flags=re.IGNORECASE)
        cleaned = pattern.sub("[NAME]", cleaned)
    return cleaned


def anonymize_text(
    text: str,
    rules: Iterable[AnonymizationRule] = DEFAULT_RULES,
    name_blocklist: Sequence[str] | None = None,
) -> str:
    cleaned = text
    for rule in rules:
        cleaned = re.sub(rule.pattern, f"[{rule.label}]", cleaned, flags=re.IGNORECASE)
    if name_blocklist:
        cleaned = apply_blocklist(cleaned, name_blocklist)
    return cleaned
