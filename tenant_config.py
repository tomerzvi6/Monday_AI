"""
Tenant Configuration — White-Label Support
============================================
A single config module that controls branding, plans, payment links,
and agent persona for any tenant.  Swap this file (or load from a DB /
env-var / JSON) to re-skin the entire agent for a different customer.

This demonstrates a **White-Label Architecture** where the AI Sales
Agent can be deployed for monday.com *or* any partner / reseller
with their own branding — without touching application code.

Usage:
    from tenant_config import TENANT
    print(TENANT.brand_name)       # "monday.com"
    print(TENANT.plans["pro"])     # Plan(name='Pro', ...)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from models import parse_team_size


# ──────────────────────────────────────────────
# Plan Definition
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class Plan:
    """A single pricing plan offered during the closing stage."""

    name: str
    price_per_seat: float
    currency: str = "USD"
    billing_cycle: str = "month"
    features: tuple[str, ...] = ()
    stripe_link: str = ""
    trial_days: int = 14


# ──────────────────────────────────────────────
# Tenant Definition
# ──────────────────────────────────────────────
@dataclass
class TenantConfig:
    """All configurable knobs for a single tenant / brand deployment.

    Swap this object to white-label the agent for any partner.
    """

    # ── Branding ──────────────────────────────
    brand_name: str = "monday.com"
    brand_tagline: str = "The Work OS that powers teams to run projects and workflows with confidence"
    agent_name: str = "Mo"
    logo_url: str = (
        "https://dapulse-res.cloudinary.com/image/upload/f_auto,q_auto/"
        "remote_mondaycom_static/img/monday-logo-x2.png"
    )
    primary_color: str = "#6c22bd"
    accent_color: str = "#ff3d57"
    copyright_year: str = "2026"

    # ── Agent Persona ─────────────────────────
    agent_personality: str = (
        "sharp, slightly humorous, and highly capable B2B sales representative"
    )
    agent_greeting: str = (
        "👋 Hey there! Welcome to **{brand_name}**!\n\n"
        "I'm **{agent_name}**, your personal AI sales concierge. "
        "I'll help you find the perfect workspace setup for your team "
        "— it only takes a minute.\n\n"
        "To get started, could you tell me **what industry** you're in?"
    )

    # ── Qualification Thresholds ──────────────
    smb_max_team_size: int = 50
    mid_market_max_team_size: int = 500
    # Above mid_market_max → Enterprise

    # ── Plans ─────────────────────────────────
    plans: dict[str, Plan] = field(default_factory=lambda: {
        "basic": Plan(
            name="Basic",
            price_per_seat=9,
            features=(
                "Unlimited boards", "200+ templates",
                "Up to 5GB storage", "1 week activity log",
            ),
            stripe_link="https://buy.stripe.com/test_monday_basic",
        ),
        "standard": Plan(
            name="Standard",
            price_per_seat=12,
            features=(
                "Everything in Basic",
                "Timeline & Gantt views", "Calendar view",
                "Guest access", "250 automations/month",
            ),
            stripe_link="https://buy.stripe.com/test_monday_standard",
        ),
        "pro": Plan(
            name="Pro",
            price_per_seat=19,
            features=(
                "Everything in Standard",
                "Private boards", "Chart view",
                "Time tracking", "25K automations/month",
                "Formula column",
            ),
            stripe_link="https://buy.stripe.com/test_monday_pro",
        ),
        "enterprise": Plan(
            name="Enterprise",
            price_per_seat=0,  # custom pricing
            features=(
                "Everything in Pro",
                "Enterprise-grade security (SAML SSO, SCIM)",
                "Advanced reporting & analytics",
                "Multi-level permissions", "Tailored onboarding",
                "Premium 24/7 support", "99.9% SLA uptime",
            ),
            stripe_link="",  # routed to sales team
        ),
    })

    # ── Demo Video Links ──────────────────────
    demo_links: dict[str, str] = field(default_factory=lambda: {
        "CRM": "https://www.youtube.com/watch?v=fdYzKjken2I",
        "Project Management": "https://www.youtube.com/watch?v=fDhe_sexzSI",
        "General": "https://www.youtube.com/watch?v=EKiOeLSxDBA",
    })

    # ── Conversation Guardrails ───────────────
    max_qualification_turns: int = 12  # stuck-loop breaker
    qualification_timeout_msg: str = (
        "I've been asking a few questions — let me just make sure I have "
        "what I need.  Could you quickly confirm your **industry**, "
        "**team size**, and the **main use-case** you'd like to solve?"
    )

    # ── API / Integration ─────────────────────
    monday_api_url: str = "https://api.monday.com/v2"
    monday_api_version: str = "2024-10"

    # ── Helper methods ────────────────────────
    def recommend_plan(self, team_size_raw: str) -> Plan:
        """Pick the best plan based on team size tier.

        Returns the Plan object best suited for the lead's tier.
        """
        size = self._parse_team_size(team_size_raw)

        if size <= self.smb_max_team_size:
            return self.plans["standard"]
        elif size <= self.mid_market_max_team_size:
            return self.plans["pro"]
        else:
            return self.plans["enterprise"]

    def get_tier(self, team_size_raw: str) -> str:
        """Classify into SMB / Mid-Market / Enterprise."""
        size = self._parse_team_size(team_size_raw)
        if size <= self.smb_max_team_size:
            return "SMB"
        elif size <= self.mid_market_max_team_size:
            return "Mid-Market"
        return "Enterprise"

    def render_greeting(self) -> str:
        """Return the greeting with brand tokens replaced."""
        return self.agent_greeting.format(
            brand_name=self.brand_name,
            agent_name=self.agent_name,
        )

    @staticmethod
    def _parse_team_size(raw: str) -> int:
        """Best-effort integer extraction from free-text team size."""
        return parse_team_size(raw)


# ──────────────────────────────────────────────
# Default Tenant (monday.com)
# ──────────────────────────────────────────────
TENANT = TenantConfig()

# Override from env-vars for quick re-branding (e.g. in Docker)
if os.getenv("TENANT_BRAND_NAME"):
    TENANT.brand_name = os.getenv("TENANT_BRAND_NAME", TENANT.brand_name)
    TENANT.agent_name = os.getenv("TENANT_AGENT_NAME", TENANT.agent_name)
    TENANT.logo_url = os.getenv("TENANT_LOGO_URL", TENANT.logo_url)
    TENANT.primary_color = os.getenv("TENANT_PRIMARY_COLOR", TENANT.primary_color)
    TENANT.accent_color = os.getenv("TENANT_ACCENT_COLOR", TENANT.accent_color)
