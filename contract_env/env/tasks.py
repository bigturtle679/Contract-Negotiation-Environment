from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from contract_env.env.models import Reward
    from contract_env.env.models import Action


class NegotiationTask(BaseModel):
    id: str
    name: str
    contract_text: str
    clause_type: str
    risk_keywords: List[str]  # ✅ FIXED (was tuple before)
    safe_keywords: List[str]
    expected_safe_edit: str
    risk_level: str
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
    grader: str
    
    def get_grader(self) -> Callable[[NegotiationTask, str, Any, str], Reward]:
        """Resolve grader function by name from the graders module."""
        # Lazy import to avoid circular dependency
        from contract_env.env.graders import TASK_GRADERS
        
        if self.id not in TASK_GRADERS:
            raise ValueError(f"No grader found for task {self.id}. Available tasks: {list(TASK_GRADERS.keys())}")
        return TASK_GRADERS[self.id]
    
    def has_grader(self) -> bool:
        """Check if this task has a valid grader function."""
        try:
            self.get_grader()
            return True
        except (ValueError, KeyError):
            return False


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
        grader="grade_easy",
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
        grader="grade_medium",
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
            "no additional obligations beyond industry practice",
            "unlimited change orders",
        ],
        clause_type_weight=1.15,
        industry_context="professional_services",
        opponent_opening=[
            "[Counterparty] Unlimited changes are standard in agile delivery."
        ],
        grader="grade_hard",
    ),
]

# ============ GRADED TASK VALIDATION ============
def get_graded_tasks() -> list[NegotiationTask]:
    """Return list of all tasks that have graders explicitly configured."""
    graded = [task for task in TASKS if task.grader]
    return graded

def count_graded_tasks() -> int:
    """Return the number of tasks with graders."""
    return len(get_graded_tasks())

def validate_all_tasks_have_graders() -> bool:
    """Validate that all tasks have grader functions accessible."""
    from contract_env.env.graders import TASK_GRADERS
    
    graded_count = count_graded_tasks()
    if graded_count < 3:  # Must have at least 3 graded tasks
        return False
    
    # Check that all task IDs with graders are in TASK_GRADERS registry
    for task in TASKS:
        if task.grader and task.id not in TASK_GRADERS:
            return False
    return True

# Metadata - just count tasks with grader field set
GRADED_TASK_IDS = [task.id for task in TASKS if task.grader]
GRADED_TASK_NAMES = [task.name for task in TASKS if task.grader]
NUM_GRADED_TASKS = len(GRADED_TASK_IDS)