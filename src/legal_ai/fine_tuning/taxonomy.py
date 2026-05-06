"""Mapping from CUAD label names to the project risk taxonomy."""

from __future__ import annotations

CUAD_TO_RISK_CATEGORY: dict[str, str] = {
    "Liquidated Damages": "penalty",
    "Cap On Liability": "liability_cap",
    "Uncapped Liability": "liability_cap",
    "Exclusivity": "exclusivity",
    "Non-Compete": "exclusivity",
    "Renewal Term": "auto_renewal",
    "Auto Renewal": "auto_renewal",
    "Termination For Convenience": "unilateral_termination",
    "Notice Period To Terminate Renewal": "unilateral_termination",
    "Governing Law": "unfavorable_jurisdiction",
    "IP Ownership Assignment": "ip_assignment",
    "Joint IP Ownership": "ip_assignment",
    "License Grant": "ip_assignment",
    "Non-Disclosure Agreement": "confidentiality",
    "Confidentiality": "confidentiality",
    "Change Of Control": "change_control",
    "Price Restrictions": "change_control",
    "Most Favored Nation": "change_control",
    "Insurance": "data_protection",
}


def normalize_cuad_label(label: str) -> str:
    """Normalize a CUAD column / answer label for matching."""

    cleaned = (label or "").strip().lower()
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    return " ".join(cleaned.split())


def map_cuad_label(label: str) -> str | None:
    """Map a raw CUAD label to a project risk category, or None if unknown."""

    target = normalize_cuad_label(label)
    for key, value in CUAD_TO_RISK_CATEGORY.items():
        if normalize_cuad_label(key) == target:
            return value
    return None
