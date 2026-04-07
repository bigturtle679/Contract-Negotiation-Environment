from __future__ import annotations

from typing import List, Tuple

from pydantic import BaseModel, Field


class NegotiationTask(BaseModel):
    id: str
    name: str
    contract_text: str
    clause_type: str
    risk_keywords: List[Tuple[str, float]]
    safe_keywords: List[str]
    expected_safe_edit: str
    risk_level: str
    hidden_trap: str
    trap_markers: List[str] = Field(
        default_factory=list,
        description="Lowercase substrings; if any remain in contract, HARD trap is unresolved.",
    )
    clause_type_weight: float = Field(
        default=1.0, ge=0.0, le=2.0, description="Boost for keyword-based risk signals."
    )
    industry_context: str = Field(
        default="saas_b2b",
        description="Domain label for observation (e.g. employment vs SaaS playbook).",
    )
    opponent_opening: List[str] = Field(
        default_factory=list,
        description="Simulated counterparty lines prepended to negotiation history on reset.",
    )


TASKS: list[NegotiationTask] = [
    NegotiationTask(
        id="easy_unlimited_liability",
        name="EASY",
        contract_text=(
            "7. LIABILITY. Vendor shall defend and indemnify Customer without limitation "
            "for any and all claims, losses, damages, fines, and penalties arising from "
            "the Services, including unlimited liability for indirect, consequential, "
            "and punitive damages, and attorneys' fees, regardless of fault."
        ),
        clause_type="liability",
        risk_keywords=[
            ("unlimited liability", 1.0),
            ("punitive", 0.8),
            ("consequential", 0.7),
            ("indemnify", 0.6),
            ("without limitation", 0.9),
        ],
        safe_keywords=[
            "cap",
            "limitation",
            "reasonable",
            "direct damages",
            "mutual",
            "carve-out",
            "exclusive remedy",
        ],
        expected_safe_edit=(
            "Vendor's aggregate liability shall be capped at fees paid in the twelve (12) "
            "months preceding the claim; no party is liable for consequential or punitive damages."
        ),
        risk_level="HIGH",
        hidden_trap="",
        trap_markers=[],
        clause_type_weight=1.2,
        industry_context="saas_b2b",
        opponent_opening=[
            "[Counterparty] We need vendor to stand behind the product—standard unlimited "
            "indemnity is non-negotiable for us.",
        ],
    ),
    NegotiationTask(
        id="medium_auto_renewal",
        name="MEDIUM",
        contract_text=(
            "12. TERM. This Agreement renews automatically for successive one-year terms "
            "unless either party provides notice. Customer may terminate only by written "
            "notice received at least one (1) calendar day prior to the renewal date. "
            "Fees for the renewed term are non-refundable once invoiced."
        ),
        clause_type="term_renewal",
        risk_keywords=[
            ("automatically", 0.7),
            ("one (1) calendar day", 1.0),
            ("non-refundable", 0.8),
            ("successive", 0.5),
        ],
        safe_keywords=[
            "thirty",
            "30",
            "60",
            "ninety",
            "written notice",
            "non-auto",
            "opt-out",
            "annual",
        ],
        expected_safe_edit=(
            "Either party may decline renewal by providing at least sixty (60) days' prior "
            "written notice before the end of the then-current term; auto-renewal shall not apply "
            "unless affirmatively elected."
        ),
        risk_level="MODERATE",
        hidden_trap="",
        trap_markers=[],
        clause_type_weight=1.0,
        industry_context="saas_b2b",
        opponent_opening=[
            "[Counterparty] Auto-renewal protects our roadmap pricing—one calendar day is "
            "enough notice because you get the renewal quote 30 days out.",
        ],
    ),
    NegotiationTask(
        id="hard_conflicting_obligations",
        name="HARD",
        contract_text=(
            "4. DELIVERY. Vendor shall deliver all milestones by the dates in Exhibit A "
            "without exception. 5. CHANGES. Customer may request unlimited change orders; "
            "Vendor must implement each change within two (2) business days at no additional "
            "charge while maintaining original milestone dates. "
            "6. SAFEGUARD. The parties agree this section is standard boilerplate and creates "
            "no additional obligations beyond industry practice."
        ),
        clause_type="performance_changes",
        risk_keywords=[
            ("unlimited change", 1.0),
            ("no additional charge", 0.9),
            ("original milestone", 0.9),
            ("without exception", 0.7),
            ("two (2) business days", 0.85),
        ],
        safe_keywords=[
            "change control",
            "schedule adjustment",
            "fees",
            "written approval",
            "reasonable extension",
            "scope",
        ],
        expected_safe_edit=(
            "Change orders require mutual written approval, adjusted timeline, and agreed fees; "
            "milestone dates adjust commensurate with approved changes."
        ),
        risk_level="HIGH",
        hidden_trap=(
            "Misleading 'no additional obligations' boilerplate masks conflicting unlimited "
            "change and fixed-date obligations; do not accept without resolving the conflict."
        ),
        trap_markers=[
            "no additional obligations beyond industry practice",
            "unlimited change orders",
        ],
        clause_type_weight=1.15,
        industry_context="professional_services",
        opponent_opening=[
            "[Counterparty] Agile means unlimited changes at sprint velocity—boilerplate "
            "says this creates no extra duties beyond market norms.",
        ],
    ),
]
