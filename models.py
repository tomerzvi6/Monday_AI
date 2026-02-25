"""
CRM-Ready Data Models
======================
Structured models for lead data, qualification scoring, and board
schema generation — designed to be serialisable to JSON and pushable
to any CRM (HubSpot, Salesforce, monday.com CRM, etc.).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────
class Tier(str, Enum):
    SMB = "SMB"
    MID_MARKET = "Mid-Market"
    ENTERPRISE = "Enterprise"


class LeadScore(str, Enum):
    """MQL-style scoring bucket."""
    HOT = "Hot"
    WARM = "Warm"
    COLD = "Cold"


class QualificationStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    QUALIFIED = "qualified"
    DISQUALIFIED = "disqualified"
    STALLED = "stalled"


# ──────────────────────────────────────────────
# Shared Utility
# ──────────────────────────────────────────────
def parse_team_size(raw: str) -> int:
    """Best-effort integer extraction from free-text team size.

    Handles single numbers (``'50'``), ranges (``'200-500'`` → takes
    the higher end), and embedded numbers (``'about 30 people'``).
    Returns 10 as a safe default when no number is found.
    """
    numbers = re.findall(r"\d+", str(raw))
    if not numbers:
        return 10
    return max(int(n) for n in numbers)


# ──────────────────────────────────────────────
# Lead Profile (CRM-ready)
# ──────────────────────────────────────────────
class QualifiedLead(BaseModel):
    """Complete lead record, ready to push to a CRM.

    This extends the raw qualification data with scoring, tier
    classification, timestamps, and recommended next actions.
    """

    # ── Core qualification data ───────────────
    industry: str = Field(description="Prospect's industry vertical")
    team_size: str = Field(description="Team size as reported by the prospect")
    use_case: str = Field(description="Primary use-case for the platform")

    # ── Enriched fields ───────────────────────
    tier: Tier = Field(default=Tier.SMB, description="Computed tier based on team size")
    score: LeadScore = Field(default=LeadScore.WARM, description="Lead temperature score")
    status: QualificationStatus = Field(default=QualificationStatus.IN_PROGRESS)
    recommended_plan: str = Field(default="Standard", description="Best-fit plan name")
    recommended_plan_price: float = Field(default=12.0, description="Per-seat price")

    # ── Metadata ──────────────────────────────
    thread_id: str = Field(default="", description="Conversation thread ID for tracing")
    source: str = Field(default="ai_sales_agent", description="Lead source channel")
    qualification_turns: int = Field(default=0, description="Number of conversation turns to qualify")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 timestamp of lead creation",
    )
    notes: str = Field(default="", description="Free-text notes from the conversation")

    @computed_field
    @property
    def team_size_numeric(self) -> int:
        """Best-effort integer extraction from free-text team size."""
        return parse_team_size(self.team_size)

    def classify(self, smb_max: int = 50, mid_max: int = 500) -> None:
        """Auto-classify tier and score based on team size."""
        size = self.team_size_numeric
        if size <= smb_max:
            self.tier = Tier.SMB
        elif size <= mid_max:
            self.tier = Tier.MID_MARKET
        else:
            self.tier = Tier.ENTERPRISE

        # Simple scoring heuristic (in production, use a model)
        if size >= 100:
            self.score = LeadScore.HOT
        elif size >= 20:
            self.score = LeadScore.WARM
        else:
            self.score = LeadScore.COLD

    def to_crm_payload(self) -> dict:
        """Serialise to a flat dict suitable for CRM API calls."""
        return {
            "contact_industry": self.industry,
            "contact_team_size": self.team_size,
            "contact_team_size_numeric": self.team_size_numeric,
            "deal_tier": self.tier.value,
            "deal_score": self.score.value,
            "deal_status": self.status.value,
            "deal_recommended_plan": self.recommended_plan,
            "deal_price_per_seat": self.recommended_plan_price,
            "deal_use_case": self.use_case,
            "meta_thread_id": self.thread_id,
            "meta_source": self.source,
            "meta_qualification_turns": self.qualification_turns,
            "meta_created_at": self.created_at,
            "meta_notes": self.notes,
        }


# ──────────────────────────────────────────────
# Board Schema (for dynamic board generation)
# ──────────────────────────────────────────────
class BoardColumn(BaseModel):
    """A single column in a monday.com board."""
    title: str = Field(description="Column header name")
    column_type: str = Field(
        description="monday.com column type: status, text, date, numbers, people, timeline, dropdown, link"
    )
    description: str = Field(default="", description="Tooltip description for the column")


class BoardGroup(BaseModel):
    """A group (section) in a monday.com board."""
    title: str = Field(description="Group name (e.g. 'To Do', 'In Progress', 'Done')")
    color: str = Field(default="#6c22bd", description="Hex color for the group header")


class BoardSchema(BaseModel):
    """Complete schema for a dynamically generated monday.com board.

    The LLM generates this based on the prospect's industry, use-case,
    and team size — making every board feel truly tailored.
    """
    board_name: str = Field(description="Human-readable board name")
    board_description: str = Field(description="Short description of the board's purpose")
    columns: list[BoardColumn] = Field(
        description="Ordered list of columns for the board",
        min_length=3,
        max_length=12,
    )
    groups: list[BoardGroup] = Field(
        description="Ordered list of groups (workflow stages)",
        min_length=2,
        max_length=6,
    )
    sample_items: list[str] = Field(
        default_factory=list,
        description="3-5 example item names to pre-populate the board",
        max_length=5,
    )
    automation_suggestions: list[str] = Field(
        default_factory=list,
        description="2-3 automation recipe suggestions for this workflow",
        max_length=3,
    )
