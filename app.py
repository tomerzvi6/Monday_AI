"""
monday.com AI Sales Concierge — Streamlit Chat UI 
========================================================
White-labeled chat interface.

Launch with:
    streamlit run app.py
"""

from __future__ import annotations

import uuid

import streamlit as st

from agent_backend import run_agent
from tenant_config import TENANT

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
AI_AVATAR = "🟣"
USER_AVATAR = "👤"

# ──────────────────────────────────────────────
# Page Configuration (uses tenant branding)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title=f"{TENANT.brand_name} — AI Sales Concierge",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS (uses tenant colours)
# ──────────────────────────────────────────────
st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 2rem; }}

    .hero-banner {{
        background: linear-gradient(135deg, {TENANT.primary_color} 0%, {TENANT.accent_color} 100%);
        border-radius: 12px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.2rem;
        text-align: center;
    }}
    .hero-banner h1 {{
        color: #ffffff;
        font-size: 1.9rem;
        font-weight: 700;
        margin: 0;
    }}
    .hero-banner p {{
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin: 0.3rem 0 0 0;
    }}

    section[data-testid="stSidebar"] {{
        background: #1f1f3d;
    }}
    section[data-testid="stSidebar"] * {{
        color: #e0e0f0 !important;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.12);
    }}

    .feature-card {{
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 10px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.7rem;
    }}
    .feature-card h4 {{
        margin: 0 0 0.25rem 0;
        font-size: 0.95rem;
    }}
    .feature-card p {{
        margin: 0;
        font-size: 0.82rem;
        opacity: 0.78;
    }}

    .tech-pill {{
        display: inline-block;
        background: rgba(108,34,189,0.35);
        border: 1px solid rgba(108,34,189,0.55);
        border-radius: 20px;
        padding: 0.2rem 0.65rem;
        font-size: 0.72rem;
        margin: 0.15rem 0.2rem;
    }}

    /* ── Sidebar logo contrast ─────────── */
    section[data-testid="stSidebar"] img {{
        background: #ffffff;
        border-radius: 10px;
        padding: 0.6rem 1rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Sidebar — Branding + Features
# ──────────────────────────────────────────────
with st.sidebar:
    st.image(TENANT.logo_url, width=180)
    st.markdown("---")

    st.markdown(f"#### 🚀 AI Sales Concierge")
    st.markdown(
        f"<p style='font-size:0.88rem; opacity:0.8;'>"
        f"Your autonomous Go-To-Market assistant that qualifies, provisions, "
        f"and closes — in a single conversation.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("##### What this agent does")

    st.markdown(
        '<div class="feature-card">'
        "<h4>🎯 Autonomous Qualification</h4>"
        "<p>Asks smart, one-at-a-time questions to learn your industry, "
        "team size, and primary use-case.</p>"
        "</div>"
        '<div class="feature-card">'
        "<h4>🎨 AI-Designed Board Schema</h4>"
        "<p>Uses GPT to design custom columns, groups, and automations "
        "tailored to your exact workflow.</p>"
        "</div>"
        '<div class="feature-card">'
        "<h4>⚡ Instant Workspace Provisioning</h4>"
        "<p>Creates a tailored board via the live API — "
        "configured for your exact workflow.</p>"
        "</div>"
        '<div class="feature-card">'
        "<h4>💳 Tier-Based Closing</h4>"
        "<p>SMB gets self-serve checkout. Enterprise gets routed to "
        "a dedicated account executive.</p>"
        "</div>"
        '<div class="feature-card">'
        "<h4>🧠 Objection Handling</h4>"
        "<p>Addresses pricing, integrations, and feature questions on the fly — "
        "then smoothly steers back on track.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("##### Built with")
    st.markdown(
        '<div style="text-align:center; margin-top:0.3rem;">'
        '<span class="tech-pill">LangGraph</span>'
        '<span class="tech-pill">GPT-4o-mini</span>'
        '<span class="tech-pill">Pydantic</span>'
        '<span class="tech-pill">Streamlit</span>'
        f'<span class="tech-pill">{TENANT.brand_name} API</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    if st.button("🔄  Start New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = uuid.uuid4().hex
        st.rerun()

    st.markdown(
        f"<p style='text-align:center; font-size:0.72rem; opacity:0.5; "
        f"margin-top:1.5rem;'>© {TENANT.copyright_year} {TENANT.brand_name} — Demo Agent</p>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# Session State Initialisation
# ──────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex

if "messages" not in st.session_state:
    st.session_state.messages = []

# ──────────────────────────────────────────────
# Welcome Greeting (from tenant config)
# ──────────────────────────────────────────────
if not st.session_state.messages:
    st.session_state.messages.append(
        {"role": "assistant", "content": TENANT.render_greeting()}
    )

# ──────────────────────────────────────────────
# Header Banner
# ──────────────────────────────────────────────
st.markdown(
    f'<div class="hero-banner">'
    f"<h1>{TENANT.brand_name} — AI Sales Concierge</h1>"
    f"<p>Qualify · Design · Provision · Close — all in one conversation</p>"
    f"</div>",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Render Chat History
# ──────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = AI_AVATAR if msg["role"] == "assistant" else USER_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ──────────────────────────────────────────────
# Chat Input & Response Handling
# ──────────────────────────────────────────────
user_input = st.chat_input("Type your message here…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar=AI_AVATAR):
        with st.spinner(f"{TENANT.agent_name} is thinking…"):
            response = run_agent(user_input, st.session_state.thread_id)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
