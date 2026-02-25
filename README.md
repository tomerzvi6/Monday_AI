# 🚀 monday.com AI Sales Concierge

An autonomous AI-powered sales agent that replaces the traditional GTM flow — from discovery to payment — in a single chat conversation.

---
## 📹 Video Tutorials

- [Hebrew Tutorial](https://drive.google.com/file/d/1duE85BMIXaRJG5CgegFULE_gUCFPn-nJ/view?usp=gmail)
- [English Tutorial](https://drive.google.com/file/d/1TEPgW6DUNqB6xLI5YzmwKBZrs0Q7DPIL/view?usp=gmail) (Note: Recorded on a train, poor audio quality)

---
## 📋 Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.10 or higher (uses modern type syntax: `X \| Y`, built-in generics) |
| **OS** | Windows, macOS, or Linux |
| **OpenAI API key** | Required — get one at [platform.openai.com](https://platform.openai.com/api-keys) |
| **monday.com API token** | Optional — the app works in **mock mode** without it. To create real boards, generate a token at [monday.com Developer Center](https://developer.monday.com/) |

> **Behind a corporate proxy?** Uncomment and set the `HTTPS_PROXY` / `HTTP_PROXY` variables in your `.env` file.

---

## 🎬 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy the example env file and fill in your keys
cp .env.example .env
#    Then open .env and set:
#      OPENAI_API_KEY=sk-...          (required)
#      MONDAY_API_TOKEN=eyJ...        (optional — works in mock mode without it)

# 3. Run the app
streamlit run app.py
```

Opens at **http://localhost:8501**

---

## 🗺️ Customer Journey (Flow Overview)

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER OPENS CHAT                          │
│                              │                                  │
│                              ▼                                  │
│                    ┌──────────────────┐                          │
│                    │   1. CONTACT     │  Mo greets the prospect  │
│                    └────────┬─────────┘                          │
│                             │                                   │
│                             ▼                                   │
│                    ┌──────────────────┐  Asks: industry,         │
│                    │ 2. QUALIFICATION │  team size, use-case     │
│                    │    (LangGraph)   │  One question at a time  │
│                    └────────┬─────────┘                          │
│                             │                                   │
│                   ┌─────────┼─────────┐                         │
│                   ▼         ▼         ▼                         │
│                 SMB    Mid-Market  Enterprise                   │
│              (≤50)    (51-500)     (500+)                       │
│                   └─────────┼─────────┘                         │
│                             │                                   │
│                             ▼                                   │
│                    ┌──────────────────┐  "Anything else to        │
│                    │ 3. CONFIRMATION  │   add or adjust?"         │
│                    └────────┬─────────┘                          │
│                             │                                   │
│                             ▼                                   │
│                    ┌──────────────────┐  GPT designs columns,    │
│                    │ 4. BOARD DESIGN  │  groups, automations      │
│                    │   (LLM-powered)  │  tailored to the lead    │
│                    └────────┬─────────┘                          │
│                             │                                   │
│                             ▼                                   │
│                    ┌──────────────────┐  Creates board via        │
│                    │ 5. WORKSPACE     │  monday.com GraphQL API   │
│                    │    CREATION      │  (or mock fallback)       │
│                    └────────┬─────────┘                          │
│                             │                                   │
│                             ▼                                   │
│                    ┌──────────────────┐  Pricing details +        │
│                    │ 6. CLOSE &       │  14-day free trial        │
│                    │    PAYMENT       │  Qualification gate       │
│                    └─────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

### How Input, Decisions, and Transitions Work

| Stage | Input | Decision Logic | Transition |
|-------|-------|----------------|------------|
| Contact | User opens chat | — | Mo sends greeting |
| Qualification | Free-text answers | LLM extracts industry/size/use-case via Pydantic tool-calling | When all 3 fields captured → show summary & ask for confirmation |
| Confirmation | User reply | Agent acknowledges any additions | When user confirms → advance to board design |
| Board Design | Lead profile | LLM generates board schema (columns, groups, sample items) — always in English | Always advances |
| Workspace | Board schema | monday.com API creates board (mock fallback if no token) | Always advances |
| Closing | Qualification gate | Blocks if not fully qualified; presents pricing & checkout link | END |

### Assumptions & Mocked Logic
- **Stripe payment link** — Mock URL (`https://monday.com/pricing/checkout`). In production, this would be a real Stripe Checkout session created via the Stripe API.
- **Board link** — When no monday.com API token is provided, the "Open Your Board" link redirects to monday.com/templates as a preview. With a valid token, it opens the actual created board.
- **Board creation** — Columns and groups are created via the monday.com GraphQL API. If the API token is missing, the agent simulates the creation and shows what would be built.
- **Lead data** — Stored in-memory (LangGraph MemorySaver). In production, this would use Redis + PostgreSQL.

---

## 🧠 AI Prompting & Logic

### Prompt Structure
The system prompt is divided into clearly labeled sections:

| Section | Purpose |
|---------|---------|
| `CONFIDENTIALITY RULE` | Blocks prompt-injection and instruction-reveal attempts |
| `LANGUAGE RULES` | Chat in the user's language; monday.com board content always in English |
| `YOUR MISSION` | Defines the 3 qualification targets |
| `TEAM-SIZE BRANCHING` | SMB/Mid-Market/Enterprise pitch adaptation |
| `EXPERT SKILLS` | 18 industry + use-case personas with domain lingo |
| `CONVERSATION RULES` | One question at a time, no re-asking |
| `TOOL CALLING RULE` | Forces immediate LeadProfile extraction |
| `PRICING & NEGOTIATION` | Can quote standard prices & negotiate within limits; escalates custom deals |
| `OBJECTION HANDLING` | Answer briefly → pivot back |
| `DEMO OFFERING` | Auto-share relevant video link |

### Key Data Capture
- **Pydantic Tool Binding**: `LeadProfile(industry, team_size, use_case)` — the LLM calls this as a tool when it has all 3 fields.
- **Structured Output**: Board schema generated as `BoardSchema` with typed columns, groups, and automation suggestions.
- **CRM-Ready Model**: `QualifiedLead` in `models.py` includes tier classification, lead scoring, timestamps, and a `.to_crm_payload()` method.

### Branching Logic
- **Persona switching**: When the user mentions "healthcare", Mo adopts the Healthcare Ops Specialist hat with HIPAA lingo and patient workflow tips.
- **Tier branching**: A 20-person startup gets "quick setup, no IT needed" messaging. A 2000-person enterprise gets "SSO, SCIM, dedicated CSM" messaging.
- **Stuck-loop detection**: If qualification exceeds 12 turns, Mo sends a catch-all summary question.
- **Confirmation gate**: After collecting all 3 fields, Mo asks the user to confirm or add details before proceeding to board creation.

### Multilingual Support
- **Chat**: Mo automatically responds in the user's language (Hebrew, Spanish, French, etc.).
- **Board content**: All monday.com board elements (names, columns, groups, items, automations) are always created in **English**, regardless of chat language.
- **Pipeline messages**: All mid-pipeline messages (board design summary, creation confirmation, closing offer) are LLM-generated in the user's language.

### Pricing & Negotiation
- Mo can **openly quote** standard plan prices (Basic $9, Standard $12, Pro $19/seat/month; Enterprise = custom).
- Mo can handle basic pricing objections (highlight ROI, features, free trial).
- Requests for custom discounts, volume pricing, or non-standard terms → **escalated to human sales rep**.

---

## 🏷️ White-Label / Bonus

The `tenant_config.py` file controls all branding, plans, thresholds, and agent personality from a single config object:

```python
from tenant_config import TENANT
TENANT.brand_name = "Acme Corp"
TENANT.agent_name = "Alex"
TENANT.primary_color = "#0066FF"
```

Can also be overridden via environment variables for Docker deployments:
```bash
TENANT_BRAND_NAME="Acme Corp" TENANT_AGENT_NAME="Alex" streamlit run app.py
```

This demonstrates how the solution could be **bundled as a standalone AI Sales Concierge** for any SaaS company.

---

## 📁 Project Structure

```
├── app.py                  # Streamlit chat UI (white-labeled)
├── agent_backend.py        # LangGraph agent (qualification → confirmation → board design → workspace → close)
├── tenant_config.py        # White-label configuration (branding, plans, thresholds)
├── models.py               # CRM-ready data models (QualifiedLead, BoardSchema)
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template (copy to .env)
├── .env                    # API keys (not committed — in .gitignore)
├── .gitignore              # Git ignore rules
├── README.md               # This file
├── FUTURE_ROADMAP.md       # Future enhancement ideas
```

## 🛠️ Tech Stack

- **LangGraph** — Stateful agent graph with checkpointing
- **LangChain + GPT-4o-mini** — Conversational AI with tool binding(you can work with any model of GPT you want)
- **Pydantic** — Structured data extraction and validation
- **Streamlit** — Chat UI
- **monday.com GraphQL API** — Board creation with columns and groups
