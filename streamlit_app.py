import streamlit as st
from openai import OpenAI
import os
import json

st.set_page_config(page_title="AI Tax Assistant", page_icon="💬")

# ---------------------------------------------------------
# 🔁 Session state initialization
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # chat history for Intake Agent

if "checklist" not in st.session_state:
    st.session_state.checklist = []  # hierarchical checklist from Checklist Agent

if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}  # profile (student/working, visa, W2 status)

# Memory: which snippet index per topic (RAG-like pointer, but simple counter)
if "visual_indices" not in st.session_state:
    st.session_state.visual_indices = {}

# ---------------------------------------------------------
# 🧾 Visual snippet memory (incremental / RAG-friendly)
# ---------------------------------------------------------
# In the future, each snippet could come from a RAG chunk.
VISUAL_SNIPPETS = {
    "w2_to_1040nr": [
        # Step 1: overall + Box 1
        """
# ---------------------------------------------------------
#   W-2 → FORM 1040-NR VISUAL GUIDE (STEP 1)
#   Focus: layout + Box 1 (wages)
# ---------------------------------------------------------
#  W-2 Box 1 : Wages, tips, other compensation
#  → Form 1040-NR Line 1a
#    ("Total amount from Form(s) W-2, box 1")
# ---------------------------------------------------------
        """.strip("\n"),

        # Step 2: Box 2
        """
# ---------------------------------------------------------
#   W-2 → FORM 1040-NR VISUAL GUIDE (STEP 2)
#   Focus: Box 2 (federal tax withheld)
# ---------------------------------------------------------
#  W-2 Box 2 : Federal income tax withheld
#  → Form 1040-NR Line 25a
#    ("Federal income tax withheld from Form(s) W-2")
# ---------------------------------------------------------
        """.strip("\n"),

        # Step 3: Social Security wages/tax
        """
# ---------------------------------------------------------
#   W-2 → FORM 1040-NR VISUAL GUIDE (STEP 3)
#   Focus: Boxes 3–4 (Social Security)
# ---------------------------------------------------------
#  W-2 Box 3 : Social security wages
#  W-2 Box 4 : Social security tax withheld
#  → Not entered directly on Form 1040-NR, but useful to
#    verify your Social Security records and to check for
#    excess withholding (Form 843 in special cases).
# ---------------------------------------------------------
        """.strip("\n"),

        # Step 4: Medicare wages/tax
        """
# ---------------------------------------------------------
#   W-2 → FORM 1040-NR VISUAL GUIDE (STEP 4)
#   Focus: Boxes 5–6 (Medicare)
# ---------------------------------------------------------
#  W-2 Box 5 : Medicare wages and tips
#  W-2 Box 6 : Medicare tax withheld
#  → Not entered directly on Form 1040-NR, but used to
#    verify Medicare withholding amounts.
# ---------------------------------------------------------
        """.strip("\n"),

        # Step 5: Box 12, 14, and "other" info
        """
# ---------------------------------------------------------
#   W-2 → FORM 1040-NR VISUAL GUIDE (STEP 5)
#   Focus: Box 12, Box 14, and "other" info
# ---------------------------------------------------------
#  W-2 Box 12 : Codes (D, E, G, etc. for retirement/benefits)
#    → May affect other forms (e.g., Form 8880, retirement
#      contributions) but not a single 1040-NR line by itself.
#
#  W-2 Box 14 : "Other" information (state tax, union dues, etc.)
#    → Often relevant for state returns or recordkeeping.
# ---------------------------------------------------------
        """.strip("\n"),
    ]
}


def get_next_visual_snippets(topic: str):
    """
    Return all snippets up to the current index for a topic
    and advance the index by 1 (until the end).
    This is a simple stand-in for a future RAG-based retriever.
    """
    steps = VISUAL_SNIPPETS.get(topic, [])
    if not steps:
        return []

    current_idx = st.session_state.visual_indices.get(topic, 0)

    # Clamp index
    if current_idx < 0:
        current_idx = 0
    if current_idx >= len(steps):
        current_idx = len(steps) - 1

    # Snippets to show: from 0 .. current_idx
    snippets_to_show = steps[: current_idx + 1]

    # Advance index for next time (but don't go past last)
    if current_idx < len(steps) - 1:
        st.session_state.visual_indices[topic] = current_idx + 1
    else:
        st.session_state.visual_indices[topic] = current_idx  # stay at last

    return snippets_to_show


# ---------------------------------------------------------
# 🧠 Checklist Agent helper
# ---------------------------------------------------------
def build_tax_checklist(client: OpenAI, chat_messages, user_profile: dict):
    """
    Checklist Agent:
    Uses the entire conversation + user profile to generate/update
    a hierarchical tax-filing checklist with detailed sub-items.
    """
    if not chat_messages:
        return st.session_state.checklist  # nothing to update yet

    # Turn chat messages into a plain-text conversation transcript
    convo_lines = []
    for m in chat_messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        convo_lines.append(f"{role.upper()}: {content}")
    convo_text = "\n\n".join(convo_lines)

    system_prompt = """
You are the CHECKLIST AGENT for a US tax-filing assistant.

Another agent ("Intake Agent") is chatting with the user and asking questions.
You DO NOT talk to the user directly. Your only job is to maintain a
hierarchical checklist of tax filing tasks and information the user needs,
based on:
1) The conversation so far, and
2) The user's profile (student vs working professional, visa status, W-2 status).

You MUST return ONLY valid JSON in this EXACT format:

{
  "sections": [
    {
      "heading": "Collect W-2 forms",
      "status": "pending",
      "details": [
        {
          "item": "Collect W-2 from each employer for the tax year",
          "status": "pending"
        },
        {
          "item": "Confirm employer name, address, and EIN (Box b)",
          "status": "pending"
        },
        {
          "item": "Record wages, tips, other compensation (Box 1)",
          "status": "pending"
        },
        {
          "item": "Record federal income tax withheld (Box 2)",
          "status": "pending"
        },
        {
          "item": "Capture Social Security and Medicare wages and tax (Boxes 3–6, if applicable)",
          "status": "pending"
        }
      ]
    }
  ]
}

Rules:
- Use ACTION headings that reference specific forms or steps, for example:
  * "Collect W-2 forms"
  * "Gather 1099-INT and 1099-DIV statements"
  * "Gather 1099-NEC / 1099-K for self-employment income"
  * "Collect Form 1098-T and tuition payment records"
  * "Summarize other income (interest, dividends, scholarships, etc.)"
  * "Confirm filing information on Form 1040-NR"
- Under each heading, include 3–10 detailed sub-items ("details") that describe
  concrete information to collect or verify (box numbers, amounts, payer/employer,
  dates, etc.), not vague phrases.
- Mark a detail as "done" ONLY if the conversation clearly indicates the user
  has already provided that information or completed that step. Otherwise "pending".
- The section "status" is:
  * "done" only if all or almost all of its details appear completed,
  * otherwise "pending".
- Tailor sections to the profile:
  * Students / on F-1: likely Form 1098-T, scholarship income, on-campus W-2s.
  * Working professionals: W-2, 1099 income, retirement contributions.
  * Self-employed / 1099: business income, expenses, estimated taxes, 1099-NEC/1099-K.
- Provide between 4 and 10 sections total.
- The user never clicks checkboxes; this checklist is updated only by your inference from the conversation.
- Do NOT include any explanation text outside the JSON.
""".strip()

    user_profile_text = json.dumps(user_profile, indent=2)

    checklist_resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"User profile:\n{user_profile_text}",
            },
            {
                "role": "user",
                "content": f"Conversation so far:\n\n{convo_text}",
            },
        ],
    )

    raw = checklist_resp.choices[0].message.content or ""

    # Try to parse JSON out of the response
    try:
        # In case the model adds stray text, pull out the first {...} block
        start = raw.index("{")
        end = raw.rindex("}") + 1
        json_str = raw[start:end]
        data = json.loads(json_str)
        sections = data.get("sections", [])

        normalized_sections = []
        for sec in sections:
            heading = str(sec.get("heading", "")).strip()
            if not heading:
                continue
            sec_status = str(sec.get("status", "pending")).lower()
            if sec_status not in ["done", "pending"]:
                sec_status = "pending"

            details_raw = sec.get("details", [])
            normalized_details = []
            for det in details_raw:
                text = str(det.get("item", "")).strip()
                if not text:
                    continue
                d_status = str(det.get("status", "pending")).lower()
                if d_status not in ["done", "pending"]:
                    d_status = "pending"
                normalized_details.append({"item": text, "status": d_status})

            normalized_sections.append(
                {
                    "heading": heading,
                    "status": sec_status,
                    "details": normalized_details,
                }
            )

        if normalized_sections:
            return normalized_sections
    except Exception:
        # If parsing fails, keep the existing checklist
        pass

    return st.session_state.checklist


# ---------------------------------------------------------
# 🧠 Intake Agent: main chat assistant
# ---------------------------------------------------------
INTAKE_SYSTEM_PROMPT = """
You are the INTAKE AGENT for an AI tax assistant.

Your role:
- Talk to the user in a friendly, structured way.
- Ask step-by-step questions to understand their tax situation (e.g. W-2, 1099, 1098-T).
- Explain what you’re doing and what information you need.
- Respond naturally to their questions like “I need help with W-2” or “I’m on F-1 visa”.
- DO NOT show or mention any internal checklist or multi-agent architecture.

Another hidden agent (Checklist Agent) observes this conversation and maintains a
detailed hierarchical checklist. You do NOT manage the checklist directly, you
just have a good conversation and collect information.
""".strip()


# ---------------------------------------------------------
# 🔑 OpenAI setup
# ---------------------------------------------------------
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.info("Set your OpenAI API key in the environment variable OPENAI_API_KEY.", icon="🗝️")
    client = None
else:
    client = OpenAI(api_key=openai_api_key)

# ---------------------------------------------------------
# 🏷️ Page title & description
# ---------------------------------------------------------
st.title("💬 AI Tax Assistant")
st.write(
    "This assistant walks you through US tax filing questions. "
    "As you chat, a separate agent builds a detailed tax-filing checklist for you."
)

# ---------------------------------------------------------
# 🧍 User profile (sidebar) — simple display elements, not a checklist
# ---------------------------------------------------------
with st.sidebar:
    st.header("👤 Your Tax Profile")

    user_type = st.radio(
        "I am a...",
        ("Student", "Working professional"),
        index=0,
        key="user_type_radio",
    )

    visa_status = st.radio(
        "Are you currently on a visa in the U.S.?",
        ("Yes", "No"),
        index=0,
        key="visa_status_radio",
    )

    w2_status = st.selectbox(
        "What best describes your W-2 / income status?",
        (
            "I receive a W-2 from an employer",
            "I am self-employed / 1099 contractor",
            "Both W-2 and self-employment/1099",
            "None / other",
        ),
        key="w2_status_select",
    )

    # Store in session state for the checklist agent
    st.session_state.user_profile = {
        "employment_status": user_type,
        "on_visa": visa_status,
        "w2_status": w2_status,
    }

# ---------------------------------------------------------
# 💬 Main chat UI (Intake Agent)
# ---------------------------------------------------------
if client is not None:
    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input(
        "Tell me how I can help with your taxes today (e.g., 'I need help with my W-2')."
    )
    if prompt:
        # Add user message to conversation
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Intake Agent response (streaming)
        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model="gpt-3.5-turbo",
                stream=True,
                messages=[
                    {"role": "system", "content": INTAKE_SYSTEM_PROMPT},
                    *st.session_state.messages,
                ],
            )
            response_text = st.write_stream(stream)

        # Save assistant response
        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )

        # After each user+assistant turn, call Checklist Agent to update the checklist
        st.session_state.checklist = build_tax_checklist(
            client,
            st.session_state.messages,
            st.session_state.user_profile,
        )

# ---------------------------------------------------------
# 🔘 Incremental visual help (can be W-2 or other topics)
# ---------------------------------------------------------
st.markdown("### Need more visual help with how a form box maps to 1040-NR?")

# For now we hardcode W-2, but this 'topic' could later be
# any key that comes from a RAG result.
topic = "w2_to_1040nr"

col_v1, col_v2 = st.columns([1, 2])
with col_v1:
    if st.button("🧾 Show next W-2 → 1040-NR step"):
        # This just advances the counter in memory.
        # No RAG yet, just list-based snippets.
        snippets = get_next_visual_snippets(topic)
        st.session_state[f"visual_snippets_{topic}"] = snippets

with col_v2:
    # Render whatever we have so far for this topic
    snippets = st.session_state.get(f"visual_snippets_{topic}", [])
    if snippets:
        for snip in snippets:
            st.text(snip)
    else:
        st.caption(
            "Click the button to see the first W-2 → 1040-NR mapping snippet. "
            "Each click reveals the next column/box."
        )

# ---------------------------------------------------------
# 🧾 Read-only dynamic checklist (Checklist Agent output)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    st.header("🧾 Live Tax-filing Checklist")

    if st.session_state.checklist:
        with st.expander("Checklist (auto-updated from your chat)", expanded=True):
            for section in st.session_state.checklist:
                heading = section.get("heading", "Unnamed section")
                sec_status = section.get("status", "pending").lower()
                details = section.get("details", [])

                status_emoji = "✅" if sec_status == "done" else "⏳"
                st.markdown(f"{status_emoji} **{heading}**")

                # Sub-items (details) as bullet points, read-only
                for det in details:
                    d_item = det.get("item", "")
                    d_status = det.get("status", "pending").lower()
                    d_emoji = "✅" if d_status == "done" else "⏳"
                    st.markdown(f"- {d_emoji} {d_item}")

                st.markdown("")  # spacing
    else:
        st.info(
            "As you answer questions and say things like "
            "'I need help with my W-2' or 'I have 1099 income', "
            "a detailed checklist will appear here."
        )
