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

# Store generated visual snippets per topic
if "visual_snippets" not in st.session_state:
    # { topic: [snippet_1, snippet_2, ...] }
    st.session_state.visual_snippets = {}


def generate_visual_snippet(topic: str) -> str:
    """
    Ask the ChatGPT agent (chat completions) to generate the NEXT visual snippet
    for a given topic.
    """
    existing_snippets = st.session_state.visual_snippets.get(topic, [])
    step_number = len(existing_snippets) + 1

    # Small slice of context
    recent_messages = st.session_state.messages[-8:]
    recent_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in recent_messages
    )

    user_profile_str = ", ".join(
        f"{k}={v}" for k, v in st.session_state.user_profile.items()
    )

    system_prompt = (
        "You are a tax-focused visualization assistant that explains things in "
        "short, step-by-step, code-style text blocks. Each response should look "
        "like a mini 'visual guide' with comments and separators, suitable for "
        "a monospaced font."
    )

    user_prompt = f"""
Create the NEXT step of a visual guide for the topic: "{topic}".

- Step number: {step_number}
- User profile (if helpful): {user_profile_str or "unknown"}
- Recent conversation (for context, if relevant):
{recent_text or "[no recent messages]"}

Requirements:
- Output a SINGLE code-style text block (no backticks, just plain text).
- Start with a header comment bar and step label, for example:
  # ---------------------------------------------------------
  #   <TOPIC NAME> VISUAL GUIDE (STEP {step_number})
  #   Focus: <short focus>
  # ---------------------------------------------------------
- Then add 3–8 lines of concise explanation, using comments (#) and arrows (→)
  to show where values go or how the user should think about the step.
- Keep it focused on THIS step only. Assume previous steps already appeared above.
- Keep it under ~120 words.
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",  # or whatever model you prefer
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )

    snippet = completion.choices[0].message.content.strip()
    return snippet


def get_next_visual_snippets(topic: str):
    """
    Generate the next visual snippet for `topic` via ChatGPT and
    return ALL snippets accumulated so far for that topic.
    """
    next_snippet = generate_visual_snippet(topic)

    if topic not in st.session_state.visual_snippets:
        st.session_state.visual_snippets[topic] = []

    st.session_state.visual_snippets[topic].append(next_snippet)

    return st.session_state.visual_snippets[topic]

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
