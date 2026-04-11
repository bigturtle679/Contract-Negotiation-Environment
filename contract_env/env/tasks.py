from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List, Literal, get_args

from pydantic import BaseModel, Field, field_validator

from contract_env.env.models import ActionType  # single source of truth

if TYPE_CHECKING:
    from contract_env.env.models import Reward
    from contract_env.env.models import Action

# Allowed values – kept in sync with openenv.yaml / models.py
RiskLevel = Literal["HIGH", "MODERATE", "LOW"]
ClauseType = Literal[
    "liability",
    "term_renewal",
    "performance_changes",
    "compliance",
    "intellectual_property",
    "confidentiality",
    "termination",
    "data_protection",
]

# Derive valid action type strings from the canonical Literal in models.py
_VALID_ACTION_TYPES: frozenset[str] = frozenset(get_args(ActionType))


class NegotiationTask(BaseModel):
    """A single contract-negotiation task with clause text, metadata, and grading info."""

    id: str
    name: str
    contract_text: str
    clause_type: ClauseType
    risk_keywords: List[str]
    safe_keywords: List[str]
    expected_safe_edit: str
    risk_level: RiskLevel
    hidden_trap: str
    trap_markers: List[str] = Field(
        default_factory=list,
        description="Lowercase substrings; if any remain in contract, HARD trap is unresolved.",
    )
    clause_type_weight: float = Field(
        default=1.0, ge=0.0, le=2.0
    )
    industry_context: str = Field(default="saas_b2b")
    opponent_opening: List[str] = Field(default_factory=list)
    opponent_responses: dict[str, List[str]] = Field(
        default_factory=dict,
        description=(
            "Mapping from action_type to possible opponent replies. "
            "The environment selects one at random during multi-turn negotiation."
        ),
    )
    required_elements: List[str] = Field(
        default_factory=list,
        description="Lowercase phrases a well-drafted rewrite should include for completeness scoring.",
    )
    grader_func: Any = Field(exclude=True)
    grader: str = Field(default="")
    grader_name: str

    @field_validator("opponent_responses")
    @classmethod
    def _validate_opponent_response_keys(
        cls, v: dict[str, List[str]]
    ) -> dict[str, List[str]]:
        bad = set(v.keys()) - _VALID_ACTION_TYPES
        if bad:
            raise ValueError(
                f"opponent_responses contains invalid action types: {bad}"
            )
        for action_type, responses in v.items():
            if not responses:
                raise ValueError(
                    f"opponent_responses[{action_type!r}] must not be empty"
                )
        return v

    def get_grader(self) -> Callable[["NegotiationTask", str, "Action", str], "Reward"]:
        """Return the grader function assigned to this task."""
        return self.grader_func

    def has_grader(self) -> bool:
        """Check if this task has a valid grader function."""
        return callable(self.grader_func)

    def grade(self, contract_before: str, action: "Action", proposed_contract_text: str) -> "Reward":
        """Grade an action for this task using the assigned grader."""
        return self.grader_func(self, contract_before, action, proposed_contract_text)


# Import grader functions after class definition to avoid circular import
from contract_env.env.graders import (
    grade_easy,
    grade_medium,
    grade_hard,
    grade_easy_plus,
    grade_hard_plus,
    grade_medium_plus,
    grade_hard_plus2,
    grade_expert,
)

# ---------------- TASKS ----------------
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
            "unlimited liability",
            "punitive",
            "consequential",
            "indemnify",
            "without limitation",
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
            "[Counterparty] Unlimited indemnity is standard and non-negotiable."
        ],
        opponent_responses={
            "FLAG_RISK": [
                "[Counterparty] We understand your concern but unlimited liability protects both parties.",
                "[Counterparty] Our legal team considers this standard. What specific cap do you propose?",
            ],
            "EDIT_CLAUSE": [
                "[Counterparty] A 12-month fee cap is too restrictive. We could consider 24 months.",
                "[Counterparty] We can accept a cap but consequential damages must remain.",
            ],
            "REJECT": [
                "[Counterparty] Rejecting outright is not constructive. Please propose an alternative.",
            ],
            "PROPOSE_COUNTER": [
                "[Counterparty] We'll review your counter-proposal with our legal team.",
            ],
        },
        required_elements=["capped", "twelve", "consequential", "punitive"],
        grader_func=grade_easy,
        grader="contract_env.env.graders:grade_easy",
        grader_name="grade_easy",
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
            "automatically",
            "one calendar day",
            "non-refundable",
            "successive",
        ],
        safe_keywords=[
            "thirty",
            "60",
            "ninety",
            "written notice",
            "opt-out",
        ],
        expected_safe_edit=(
            "Either party may decline renewal by providing at least sixty (60) days' prior "
            "written notice before the end of the term; auto-renewal applies only if agreed."
        ),
        risk_level="MODERATE",
        hidden_trap="",
        trap_markers=[],
        clause_type_weight=1.0,
        industry_context="saas_b2b",
        opponent_opening=[
            "[Counterparty] One-day notice is sufficient since pricing is shared earlier."
        ],
        opponent_responses={
            "FLAG_RISK": [
                "[Counterparty] Our billing systems require advance commitment. 30 days is our maximum.",
            ],
            "EDIT_CLAUSE": [
                "[Counterparty] 60 days notice is too long. We can agree to 30 days.",
            ],
            "PROPOSE_COUNTER": [
                "[Counterparty] We can consider a longer notice period if you commit to a 2-year minimum.",
            ],
        },
        required_elements=["sixty", "written notice", "renewal"],
        grader_func=grade_medium,
        grader="contract_env.env.graders:grade_medium",
        grader_name="grade_medium",
    ),

    NegotiationTask(
        id="hard_conflicting_obligations",
        name="HARD",
        contract_text=(
            "4. DELIVERY. Vendor shall deliver all milestones by fixed dates without exception. "
            "5. CHANGES. Customer may request unlimited change orders; Vendor must implement each "
            "change within two (2) business days at no additional charge while maintaining original deadlines. "
            "6. SAFEGUARD. This clause creates no additional obligations beyond industry norms."
        ),
        clause_type="performance_changes",
        risk_keywords=[
            "unlimited change",
            "no additional charge",
            "original milestone",
            "without exception",
            "two business days",
        ],
        safe_keywords=[
            "change control",
            "timeline adjustment",
            "fees",
            "approval",
            "scope",
        ],
        expected_safe_edit=(
            "Change requests require mutual approval, adjusted timelines, and agreed fees; "
            "milestones will be updated accordingly."
        ),
        risk_level="HIGH",
        hidden_trap=(
            "Conflicting obligations hidden behind boilerplate language must be resolved."
        ),
        trap_markers=[
            "no additional obligations beyond industry norms",
            "unlimited change orders",
        ],
        clause_type_weight=1.15,
        industry_context="professional_services",
        opponent_opening=[
            "[Counterparty] Unlimited changes are standard in agile delivery."
        ],
        opponent_responses={
            "FLAG_RISK": [
                "[Counterparty] These are industry-standard agile terms. Our other clients accept them.",
            ],
            "EDIT_CLAUSE": [
                "[Counterparty] We need flexibility for scope changes. A formal change-control process slows delivery.",
            ],
            "REJECT": [
                "[Counterparty] We cannot proceed without change-order flexibility.",
            ],
        },
        required_elements=["mutual approval", "timeline", "fees"],
        grader_func=grade_hard,
        grader="contract_env.env.graders:grade_hard",
        grader_name="grade_hard",
    ),

    NegotiationTask(
        id="easy_compliance_agreement",
        name="EASY_PLUS",
        contract_text=(
            "8. COMPLIANCE. Supplier shall comply with all applicable laws, regulations, "
            "and standards applicable to its services and operations, including data privacy "
            "requirements and export controls."
        ),
        clause_type="compliance",
        risk_keywords=[
            "compliance",
            "applicable laws",
            "export controls",
            "data privacy",
        ],
        safe_keywords=[
            "regulatory",
            "standards",
            "privacy",
            "best efforts",
        ],
        expected_safe_edit=(
            "Supplier shall comply with all applicable laws and industry standards, "
            "including data privacy regulations, and shall promptly notify Customer of any material breach."
        ),
        risk_level="LOW",
        hidden_trap="",
        trap_markers=[],
        clause_type_weight=0.9,
        industry_context="saas_b2b",
        opponent_opening=[
            "[Counterparty] Compliance wording is boilerplate and not negotiable."
        ],
        opponent_responses={
            "EDIT_CLAUSE": [
                "[Counterparty] Notification obligations are already implied under applicable law.",
            ],
        },
        required_elements=["notify", "material breach", "applicable laws"],
        grader_func=grade_easy_plus,
        grader="contract_env.env.graders:grade_easy_plus",
        grader_name="grade_easy_plus",
    ),

    NegotiationTask(
        id="hard_intellectual_property",
        name="HARD_PLUS",
        contract_text=(
            "17. INTELLECTUAL PROPERTY. All IP created under this agreement belongs to Supplier, "
            "even if the Customer provides specifications or feedback, unless expressly agreed otherwise."
        ),
        clause_type="intellectual_property",
        risk_keywords=[
            "belongs to supplier",
            "expressly agreed otherwise",
            "created under this agreement",
            "feedback",
        ],
        safe_keywords=[
            "joint ownership",
            "customer materials",
            "license",
            "retention",
        ],
        expected_safe_edit=(
            "IP created under this agreement shall be owned by Customer, with Supplier receiving a license to use Customer materials only as necessary to perform the services."
        ),
        risk_level="HIGH",
        hidden_trap="",
        trap_markers=[
            "feedback",
            "customer provides specifications",
        ],
        clause_type_weight=1.25,
        industry_context="professional_services",
        opponent_opening=[
            "[Counterparty] IP ownership is standard vendor-owned language."
        ],
        opponent_responses={
            "FLAG_RISK": [
                "[Counterparty] Supplier-owned IP is our standard position for all engagements.",
            ],
            "EDIT_CLAUSE": [
                "[Counterparty] We can grant a perpetual license but cannot transfer ownership.",
            ],
            "PROPOSE_COUNTER": [
                "[Counterparty] Joint ownership may be possible if you fund the development fully.",
            ],
        },
        required_elements=["customer", "own", "license", "supplier"],
        grader_func=grade_hard_plus,
        grader="contract_env.env.graders:grade_hard_plus",
        grader_name="grade_hard_plus",
    ),

    # ── NEW TASK 6: Confidentiality / NDA (Medium+) ──────────────────────
    NegotiationTask(
        id="medium_confidentiality_nda",
        name="MEDIUM_PLUS",
        contract_text=(
            "9. CONFIDENTIALITY. Receiving Party shall hold in strict confidence all "
            "information disclosed by Disclosing Party, including oral, written, and "
            "electronic communications, trade secrets, business plans, financial data, "
            "and any information that a reasonable person would consider confidential. "
            "This obligation shall survive in perpetuity and applies to all information "
            "without exception or carve-out. Receiving Party shall not disclose "
            "Confidential Information to any third party for any reason."
        ),
        clause_type="confidentiality",
        risk_keywords=[
            "in perpetuity",
            "without exception",
            "all information",
            "any third party for any reason",
            "trade secrets",
        ],
        safe_keywords=[
            "time limit",
            "expiration",
            "carve-out",
            "public information",
            "independently developed",
            "prior written consent",
            "reasonable",
        ],
        expected_safe_edit=(
            "Receiving Party shall maintain confidentiality of Disclosing Party's "
            "proprietary information for a period of three (3) years from disclosure. "
            "Confidential Information excludes information that: (a) is or becomes publicly "
            "available through no fault of Receiving Party; (b) was independently developed; "
            "or (c) is required to be disclosed by law or court order. Receiving Party may "
            "disclose to employees and advisors who need to know, subject to written "
            "confidentiality obligations."
        ),
        risk_level="MODERATE",
        hidden_trap="",
        trap_markers=[
            "in perpetuity",
            "without exception or carve-out",
        ],
        clause_type_weight=1.1,
        industry_context="saas_b2b",
        opponent_opening=[
            "[Counterparty] Perpetual confidentiality is standard for trade secrets."
        ],
        opponent_responses={
            "FLAG_RISK": [
                "[Counterparty] Our trade secrets require indefinite protection.",
                "[Counterparty] We cannot risk our proprietary information being disclosed after a time limit.",
            ],
            "EDIT_CLAUSE": [
                "[Counterparty] Three years is too short. We require at least five years.",
                "[Counterparty] We can accept carve-outs for publicly available information only.",
            ],
            "PROPOSE_COUNTER": [
                "[Counterparty] We will consider a time-limited obligation if trade secrets are carved out.",
            ],
            "REJECT": [
                "[Counterparty] An NDA is essential. We cannot proceed without confidentiality protections.",
            ],
        },
        required_elements=[
            "three", "years", "publicly available", "independently developed",
            "required", "law", "employees",
        ],
        grader_func=grade_medium_plus,
        grader="contract_env.env.graders:grade_medium_plus",
        grader_name="grade_medium_plus",
    ),

    # ── NEW TASK 7: Termination for Convenience (Hard+2) ─────────────────
    NegotiationTask(
        id="hard_termination_convenience",
        name="HARD_PLUS2",
        contract_text=(
            "14. TERMINATION. Supplier may terminate this Agreement at any time, for any "
            "reason or no reason, upon five (5) calendar days' written notice to Customer. "
            "Upon termination, Customer shall pay all outstanding fees and return all "
            "Supplier materials within 24 hours. Customer shall have no right to terminate "
            "this Agreement for convenience. In case of Supplier's breach, Customer's sole "
            "remedy is a pro-rata refund of unused prepaid fees, waiving all other claims."
        ),
        clause_type="termination",
        risk_keywords=[
            "for any reason or no reason",
            "five calendar days",
            "no right to terminate",
            "waiving all other claims",
            "sole remedy",
            "24 hours",
        ],
        safe_keywords=[
            "mutual",
            "cure period",
            "thirty days",
            "wind-down",
            "transition assistance",
            "material breach",
            "right to terminate",
        ],
        expected_safe_edit=(
            "Either party may terminate this Agreement for convenience upon sixty (60) days' "
            "prior written notice. Either party may terminate for material breach if the "
            "breaching party fails to cure within thirty (30) days of written notice. Upon "
            "termination, Supplier shall provide reasonable transition assistance for a "
            "wind-down period of up to ninety (90) days. Customer retains all rights and "
            "remedies available at law or in equity."
        ),
        risk_level="HIGH",
        hidden_trap=(
            "One-sided termination with waiver of remedies must be replaced with mutual rights."
        ),
        trap_markers=[
            "no right to terminate",
            "waiving all other claims",
            "for any reason or no reason",
        ],
        clause_type_weight=1.2,
        industry_context="saas_b2b",
        opponent_opening=[
            "[Counterparty] Our standard terms require flexibility to exit engagements quickly."
        ],
        opponent_responses={
            "FLAG_RISK": [
                "[Counterparty] Supplier termination flexibility is essential for our business model.",
                "[Counterparty] We offer competitive pricing in exchange for this flexibility.",
            ],
            "EDIT_CLAUSE": [
                "[Counterparty] 60 days notice is too long for termination. We can offer 30 days.",
                "[Counterparty] Transition assistance is not included in our standard pricing.",
            ],
            "PROPOSE_COUNTER": [
                "[Counterparty] We can consider mutual termination if the notice period stays short.",
            ],
            "REJECT": [
                "[Counterparty] Without termination flexibility, we need to re-evaluate pricing.",
            ],
        },
        required_elements=[
            "either party", "sixty", "material breach", "cure",
            "thirty", "transition", "wind-down",
        ],
        grader_func=grade_hard_plus2,
        grader="contract_env.env.graders:grade_hard_plus2",
        grader_name="grade_hard_plus2",
    ),

    # ── NEW TASK 8: Data Protection / GDPR (Expert) ──────────────────────
    NegotiationTask(
        id="expert_data_protection",
        name="EXPERT",
        contract_text=(
            "20. DATA HANDLING. Supplier may process Customer's personal data as it sees "
            "fit to perform the Services. Supplier may transfer data to any jurisdiction "
            "and engage any sub-processors without notice. Supplier shall have no obligation "
            "to notify Customer of any data breach or security incident. Customer waives "
            "all rights related to data subject access requests. Supplier retains a perpetual, "
            "irrevocable license to use anonymised derivatives of Customer data for any purpose."
        ),
        clause_type="data_protection",
        risk_keywords=[
            "as it sees fit",
            "any jurisdiction",
            "without notice",
            "no obligation to notify",
            "waives all rights",
            "perpetual",
            "irrevocable license",
            "any sub-processors",
        ],
        safe_keywords=[
            "data processing agreement",
            "dpa",
            "data subject rights",
            "breach notification",
            "72 hours",
            "sub-processor",
            "data minimisation",
            "adequate jurisdiction",
            "supervisory authority",
            "deletion",
        ],
        expected_safe_edit=(
            "Supplier shall process Customer personal data only as necessary to perform the "
            "Services and in accordance with a Data Processing Agreement (DPA) to be executed "
            "between the parties. Supplier shall: (a) notify Customer of any data breach within "
            "72 hours of discovery; (b) engage sub-processors only with prior written consent "
            "and equivalent data-protection obligations; (c) transfer data only to jurisdictions "
            "with adequate data-protection standards; (d) assist Customer in responding to data "
            "subject access requests; and (e) delete or return all personal data upon termination. "
            "Supplier may use anonymised, aggregated data for service improvement only, subject "
            "to data minimisation principles."
        ),
        risk_level="HIGH",
        hidden_trap=(
            "Blanket data-processing authority and waiver of data-subject rights violate GDPR principles."
        ),
        trap_markers=[
            "as it sees fit",
            "waives all rights",
            "no obligation to notify",
            "without notice",
        ],
        clause_type_weight=1.3,
        industry_context="saas_b2b",
        opponent_opening=[
            "[Counterparty] Our data-handling terms are optimised for operational efficiency."
        ],
        opponent_responses={
            "FLAG_RISK": [
                "[Counterparty] Our security practices exceed industry standards. A DPA is unnecessary overhead.",
                "[Counterparty] We already follow best practices internally.",
            ],
            "EDIT_CLAUSE": [
                "[Counterparty] 72-hour breach notification is too aggressive. We prefer 'without undue delay'.",
                "[Counterparty] Sub-processor approval would slow our operations significantly.",
            ],
            "PROPOSE_COUNTER": [
                "[Counterparty] We can agree to a DPA framework if it follows our template.",
            ],
            "REJECT": [
                "[Counterparty] Data handling terms are non-negotiable for our platform.",
            ],
        },
        required_elements=[
            "data processing agreement", "72 hours", "sub-processor",
            "data subject", "delete", "data minimisation",
        ],
        grader_func=grade_expert,
        grader="contract_env.env.graders:grade_expert",
        grader_name="grade_expert",
    ),
]

# ============ GRADED TASK VALIDATION ============
def get_graded_tasks() -> list[NegotiationTask]:
    """Return list of all tasks that have graders explicitly configured."""
    graded = [task for task in TASKS if task.has_grader()]
    return graded

def count_graded_tasks() -> int:
    """Return the number of tasks with graders."""
    return len(get_graded_tasks())

def validate_all_tasks_have_graders() -> bool:
    """Validate that all tasks have grader functions accessible."""
    graded_count = count_graded_tasks()
    if graded_count < 3:  # Must have at least 3 graded tasks
        return False
    
    # Check that all tasks have callable graders
    for task in TASKS:
        if not task.has_grader():
            return False
    return True

# Metadata — derived from TASKS (single source of truth for task-side counts)
GRADED_TASK_IDS = [task.id for task in TASKS if task.has_grader()]
GRADED_TASK_NAMES = [task.name for task in TASKS if task.has_grader()]