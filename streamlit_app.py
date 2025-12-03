# Fix sqlite3 issue (must be at the very top)
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except (ImportError, KeyError):
    pass

import streamlit as st
from openai import OpenAI
import os
import json
import re
from typing import Dict, List, Tuple

# RAG imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Session Memory imports
try:
    from sessionmemory import SessionMemoryManager, UserSession
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    print("⚠️ Session memory not available. Install session_memory.py for persistence.")

# OCR imports
try:
    import easyocr
    from PIL import Image
    import numpy as np
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Document processing imports
try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# ==========================================
# Configuration
# ==========================================
DB_DIRECTORY = "federal_tax_vector_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "federal_tax_documents"

st.set_page_config(page_title="AI Tax Assistant", page_icon="💬", layout="wide")

# ==========================================
# Legal Disclaimers and Privacy Notices
# ==========================================
LEGAL_DISCLAIMER = """
⚠️ **IMPORTANT DISCLAIMER**

This AI Tax Assistant is for **educational and informational purposes only**.

• I am NOT a Certified Public Accountant (CPA), tax attorney, or licensed tax professional.
• This tool does NOT constitute professional tax advice, legal advice, or financial advice.
• Tax laws are complex and vary by individual circumstances. Always consult a qualified tax professional.
• You are solely responsible for the accuracy of your tax filings.
• The developers of this tool are not liable for any errors or omissions.

By using this tool, you acknowledge and accept these terms.
"""

PRIVACY_NOTICE = """
🔒 **PRIVACY & DATA HANDLING NOTICE**

• We automatically detect and mask sensitive information (SSN, EIN, account numbers).
• Your data is processed in-session only and is NOT permanently stored.
• Uploaded documents are processed temporarily and cleared after use.
• We recommend NOT uploading documents containing actual SSNs or bank account numbers.
• For maximum security, use sample/dummy data when learning how to file.

Your privacy is important to us.
"""

# ==========================================
# Custom CSS for Banners
# ==========================================
st.markdown("""
<style>
.disclaimer-banner {
    background-color: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 5px;
    padding: 10px 15px;
    margin-bottom: 15px;
    font-size: 0.85em;
}
.privacy-banner {
    background-color: #d1ecf1;
    border: 1px solid #17a2b8;
    border-radius: 5px;
    padding: 10px 15px;
    margin-bottom: 15px;
    font-size: 0.85em;
}
.pii-warning {
    background-color: #f8d7da;
    border: 1px solid #dc3545;
    border-radius: 5px;
    padding: 10px 15px;
    margin: 10px 0;
    font-size: 0.9em;
}
.upload-warning {
    background-color: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 5px;
    padding: 8px 12px;
    margin: 5px 0;
    font-size: 0.8em;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# PII Detection and Masking
# ==========================================
class PIIHandler:
    """Handles detection and masking of Personally Identifiable Information"""
    
    PATTERNS = {
        'ssn': [
            r'\b\d{3}-\d{2}-\d{4}\b',
            r'\b\d{3}\s\d{2}\s\d{4}\b',
            r'\b\d{9}\b(?!\d)',
        ],
        'ein': [
            r'\b\d{2}-\d{7}\b',
        ],
        'bank_account': [
            r'\b\d{8,17}\b',
        ],
        'credit_card': [
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            r'\b\d{16}\b',
        ],
        'phone': [
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
        ],
        'email': [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        ],
    }
    
    MASKS = {
        'ssn': '***-**-****',
        'ein': '**-*******',
        'bank_account': '[ACCOUNT-MASKED]',
        'credit_card': '****-****-****-****',
        'phone': '(***) ***-****',
        'email': '[EMAIL-MASKED]',
    }
    
    @classmethod
    def detect_pii(cls, text: str) -> Dict[str, List[str]]:
        found = {}
        for pii_type, patterns in cls.PATTERNS.items():
            matches = []
            for pattern in patterns:
                matches.extend(re.findall(pattern, text, re.IGNORECASE))
            if matches:
                found[pii_type] = list(set(matches))
        return found
    
    @classmethod
    def mask_pii(cls, text: str) -> Tuple[str, Dict[str, int]]:
        masked_text = text
        masked_counts = {}
        
        for pii_type, patterns in cls.PATTERNS.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, masked_text, re.IGNORECASE)
                count += len(matches)
                masked_text = re.sub(pattern, cls.MASKS[pii_type], masked_text, flags=re.IGNORECASE)
            if count > 0:
                masked_counts[pii_type] = count
        
        return masked_text, masked_counts
    
    @classmethod
    def get_pii_warning(cls, detected: Dict[str, int]) -> str:
        if not detected:
            return ""
        
        warnings = ["⚠️ **Sensitive Information Detected & Masked:**"]
        
        pii_labels = {
            'ssn': 'Social Security Number(s)',
            'ein': 'Employer Identification Number(s)',
            'bank_account': 'Bank Account Number(s)',
            'credit_card': 'Credit Card Number(s)',
            'phone': 'Phone Number(s)',
            'email': 'Email Address(es)',
        }
        
        for pii_type, count in detected.items():
            label = pii_labels.get(pii_type, pii_type)
            warnings.append(f"• {count} {label} detected and masked")
        
        warnings.append("\n*Your sensitive data has been automatically protected.*")
        
        return "\n".join(warnings)

# ==========================================
# RAG Setup (Cached)
# ==========================================
@st.cache_resource
def load_vector_db():
    """Load ChromaDB vector database"""
    if os.path.exists(DB_DIRECTORY):
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        db = Chroma(
            persist_directory=DB_DIRECTORY,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        return db
    return None

@st.cache_resource
def load_ocr():
    """Load OCR model"""
    if OCR_AVAILABLE:
        return easyocr.Reader(["en"], gpu=False)
    return None

@st.cache_resource
def load_memory_manager():
    """Load session memory manager"""
    if MEMORY_AVAILABLE:
        return SessionMemoryManager()
    return None

# ==========================================
# Session Management Functions
# ==========================================
def get_or_create_session_id() -> str:
    """Get existing session ID or create new one"""
    query_params = st.query_params
    session_id = query_params.get("session_id", None)
    
    if not session_id and 'session_id' in st.session_state:
        session_id = st.session_state.session_id
    
    if not session_id:
        import uuid
        session_id = str(uuid.uuid4())
    
    return session_id

def load_session_from_memory(memory_manager, session_id: str):
    """Load session from persistent memory"""
    if not memory_manager:
        return None
    
    session = memory_manager.get_session(session_id)
    
    if not session:
        session = memory_manager.create_session()
        session.session_id = session_id
        memory_manager.save_session(session)
    
    return session

def sync_session_to_memory(memory_manager, session, user_profile: dict, 
                           checklist: list, messages: list):
    """Sync current session state to persistent memory"""
    if not memory_manager or not session:
        return
    
    # Update session with profile data
    session.citizenship_status = user_profile.get('employment_status')
    session.student_status = "Student" if user_profile.get('employment_status') == "Student" else "Working"
    
    # Calculate completions
    profile_fields = [session.citizenship_status, session.student_status]
    session.profile_completion = (sum(1 for f in profile_fields if f) / 2) * 100
    
    if checklist:
        session.checklist_completion = sum(s.get('completion', 0) for s in checklist) / len(checklist)
    
    # Save to memory
    memory_manager.save_session(session)
    memory_manager.save_checklist(session.session_id, checklist)
    
    # Save new messages
    existing_messages = memory_manager.get_conversation_history(session.session_id)
    existing_count = len(existing_messages)
    
    for msg in messages[existing_count:]:
        memory_manager.save_message(
            session.session_id,
            msg.get('role', 'user'),
            msg.get('content', '')
        )

def load_session_state_from_memory(memory_manager, session):
    """Load session state from persistent memory"""
    if not memory_manager or not session:
        return
    
    # Load conversation history
    messages = memory_manager.get_conversation_history(session.session_id)
    if messages:
        st.session_state.messages = [
            {"role": msg['role'], "content": msg['content'].split(": ", 1)[-1] if ": " in msg['content'] else msg['content']}
            for msg in messages
        ]
    
    # Load checklist
    checklist = memory_manager.get_checklist(session.session_id)
    if checklist:
        st.session_state.checklist = checklist

# ==========================================
# RAG Search Functions
# ==========================================
def rag_search(query: str, k: int = 3, doc_type: str = "all") -> str:
    """Search ChromaDB for relevant tax information"""
    db = load_vector_db()
    if not db:
        return "Tax database is not available."
    
    try:
        filter_dict = {"doc_type": doc_type} if doc_type != "all" else None
        results = db.similarity_search(query, k=k, filter=filter_dict)
        
        if not results:
            return "No relevant information found in the tax database."
        
        response = ""
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get('source_file', 'Unknown')
            form = doc.metadata.get('form_number', 'N/A')
            content = doc.page_content[:400]
            response += f"**Source {i}** - {source} (Form {form}):\n{content}...\n\n"
        
        return response
    except Exception as e:
        return f"Error searching database: {str(e)}"

def rag_search_for_visual(source_form: str, target_form: str, step: int) -> str:
    """Search for specific form mapping information for visual generation"""
    db = load_vector_db()
    if not db:
        return ""
    
    step_queries = {
        1: f"{source_form} Box 1 wages compensation {target_form} line",
        2: f"{source_form} Box 2 federal income tax withheld {target_form}",
        3: f"{source_form} Box 3 4 Social Security wages tax {target_form}",
        4: f"{source_form} Box 5 6 Medicare wages tax {target_form}",
        5: f"{source_form} Box 12 14 codes other information {target_form}",
    }
    
    query = step_queries.get(step, f"{source_form} to {target_form} mapping step {step}")
    
    try:
        results = db.similarity_search(query, k=2)
        if results:
            return "\n\n".join([
                f"IRS Reference ({doc.metadata.get('source_file', 'Unknown')}):\n{doc.page_content[:300]}"
                for doc in results
            ])
    except Exception as e:
        print(f"RAG search error: {e}")
    
    return ""

# ==========================================
# Document Text Extraction (with PII masking)
# ==========================================
def extract_text_from_file(uploaded_file):
    """Extract text from uploaded document with PII masking"""
    file_type = uploaded_file.type
    text = ""
    
    try:
        if file_type == "text/plain":
            text = uploaded_file.read().decode("utf-8", errors="ignore")
        elif file_type == "application/pdf":
            if not PDF_AVAILABLE:
                return "❌ PDF support not installed.", {}, ""
            reader = PdfReader(uploaded_file)
            text = "\n".join([page.extract_text() or "" for page in reader.pages[:10]])
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            if not DOCX_AVAILABLE:
                return "❌ DOCX support not installed.", {}, ""
            doc = Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        return f"❌ Error extracting text: {str(e)}", {}, ""
    
    # Mask PII
    masked_text, pii_counts = PIIHandler.mask_pii(text)
    warning = PIIHandler.get_pii_warning(pii_counts) if pii_counts else ""
    
    return masked_text, pii_counts, warning

def extract_text_from_image(uploaded_image):
    """Extract text from image using OCR with PII masking"""
    if not OCR_AVAILABLE:
        return "❌ OCR not installed.", {}, ""
    
    try:
        ocr_reader = load_ocr()
        image = Image.open(uploaded_image)
        image_np = np.array(image)
        results = ocr_reader.readtext(image_np)
        text = "\n".join([res[1] for res in results])
        
        # Mask PII
        masked_text, pii_counts = PIIHandler.mask_pii(text)
        warning = PIIHandler.get_pii_warning(pii_counts) if pii_counts else ""
        
        return masked_text, pii_counts, warning
    except Exception as e:
        return f"❌ OCR Error: {str(e)}", {}, ""

# ==========================================
# Session State Initialization
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "checklist" not in st.session_state:
    st.session_state.checklist = []

if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}

if "visual_indices" not in st.session_state:
    st.session_state.visual_indices = {}

if "visual_snippets" not in st.session_state:
    st.session_state.visual_snippets = {}

if "current_visual_topic" not in st.session_state:
    st.session_state.current_visual_topic = None

# NEW: Document upload state
if "uploaded_doc_text" not in st.session_state:
    st.session_state.uploaded_doc_text = None

if "uploaded_doc_name" not in st.session_state:
    st.session_state.uploaded_doc_name = None

if "uploaded_img_text" not in st.session_state:
    st.session_state.uploaded_img_text = None

if "uploaded_img_name" not in st.session_state:
    st.session_state.uploaded_img_name = None

if "search_result" not in st.session_state:
    st.session_state.search_result = None

# ==========================================
# Visual Topic Inference (RAG-Enhanced)
# ==========================================
def infer_visual_topic(client: OpenAI) -> str:
    """Infer the most relevant visual topic from conversation using RAG context"""
    recent_messages = st.session_state.messages[-10:]
    recent_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in recent_messages
    )
    
    user_profile_str = ", ".join(
        f"{k}={v}" for k, v in st.session_state.user_profile.items()
    )
    
    # Get RAG context to help with topic inference
    rag_context = ""
    if recent_text:
        rag_context = rag_search(recent_text[:200], k=1)
    
    system_prompt = """You are a routing assistant that chooses a single short topic key for a tax visualization component.
Your ONLY job is to output a machine-friendly slug for the topic.

Available topics:
- w2_to_1040nr (W-2 to Form 1040-NR for nonresidents/F-1 students)
- w2_to_1040 (W-2 to Form 1040 for US residents)
- 1098t_to_1040nr (Form 1098-T tuition to 1040-NR)
- 1098t_to_1040 (Form 1098-T to Form 1040)
- 1099int_to_1040 (1099-INT interest income)
- 1099nec_to_schedule_c (1099-NEC self-employment)
- schedule1_adjustments (Schedule 1 adjustments)
- generic_tax_visual (general guidance)

Rules:
- International students/F-1 visa → use 1040nr variants
- US citizens/residents → use 1040 variants
- Students with tuition → 1098t topics
- Self-employed → 1099nec or schedule_c
- Respond with EXACTLY ONE topic key, nothing else."""

    user_prompt = f"""
Conversation so far:
{recent_text or "[no recent messages]"}

User profile: {user_profile_str or "unknown"}

{f"Relevant IRS context: {rag_context[:200]}" if rag_context else ""}

Output the most relevant topic key:"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        topic_key = completion.choices[0].message.content.strip().lower()
        
        valid_topics = [
            "w2_to_1040nr", "w2_to_1040", "1098t_to_1040nr", "1098t_to_1040",
            "1099int_to_1040", "1099nec_to_schedule_c", "schedule1_adjustments",
            "generic_tax_visual"
        ]
        
        if topic_key not in valid_topics:
            topic_key = "w2_to_1040nr"
        return topic_key
    except Exception as e:
        print(f"Topic inference error: {e}")
        return "w2_to_1040nr"

def parse_topic(topic: str) -> dict:
    """Parse topic key into source and target forms"""
    mappings = {
        "w2_to_1040nr": {"source": "W-2", "target": "1040-NR"},
        "w2_to_1040": {"source": "W-2", "target": "1040"},
        "1098t_to_1040nr": {"source": "1098-T", "target": "1040-NR"},
        "1098t_to_1040": {"source": "1098-T", "target": "1040"},
        "1099int_to_1040": {"source": "1099-INT", "target": "1040"},
        "1099nec_to_schedule_c": {"source": "1099-NEC", "target": "Schedule C"},
        "schedule1_adjustments": {"source": "Various", "target": "Schedule 1"},
        "generic_tax_visual": {"source": "General", "target": "Tax Return"},
    }
    return mappings.get(topic, {"source": "W-2", "target": "1040-NR"})

# ==========================================
# Visual Snippet Generation (RAG-Enhanced)
# ==========================================
def generate_visual_snippet(client: OpenAI, topic: str) -> str:
    """Generate the NEXT visual snippet using RAG for accuracy"""
    existing_snippets = st.session_state.visual_snippets.get(topic, [])
    step_number = len(existing_snippets) + 1
    
    # Parse topic
    forms = parse_topic(topic)
    source_form = forms["source"]
    target_form = forms["target"]
    
    # Get RAG context for this specific step
    rag_context = rag_search_for_visual(source_form, target_form, step_number)
    
    recent_messages = st.session_state.messages[-8:]
    recent_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in recent_messages
    )
    
    user_profile_str = ", ".join(
        f"{k}={v}" for k, v in st.session_state.user_profile.items()
    )

    system_prompt = """You are a tax visualization expert. Create step-by-step visual guides showing how to map values from source tax forms to destination forms.

Your output should be a code-style text block with:
- Clear header with step number and focus
- Box-to-line mappings using arrows (→)
- Specific box numbers and line numbers from IRS documentation
- Brief explanations
- Example values where helpful

Use the IRS documentation provided to ensure accuracy."""

    user_prompt = f"""Create Step {step_number} of a visual guide for: {source_form} → {target_form}

User Profile: {user_profile_str or "unknown"}

{"IRS Documentation Reference:" + chr(10) + rag_context if rag_context else ""}

Recent conversation context:
{recent_text or "[no recent messages]"}

Requirements:
1. Start with a header block like:
   📋 {source_form} → {target_form} Mapping (Step {step_number}/5)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Focus: [specific focus for this step]

2. Show 1-2 specific box-to-line mappings with arrows (→)
3. Include brief explanation of what each value represents
4. Reference the IRS documentation if provided
5. Add example if helpful
6. End with a separator line
7. Keep under 150 words

For step {step_number}, focus on:
- Step 1: Wages/compensation (Box 1)
- Step 2: Federal tax withheld (Box 2)  
- Step 3: Social Security (Boxes 3-4)
- Step 4: Medicare (Boxes 5-6)
- Step 5: Other codes and state info (Boxes 12, 14)"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating visual: {str(e)}"

def get_next_visual_snippets(client: OpenAI, topic: str):
    """Generate and store the next visual snippet"""
    next_snippet = generate_visual_snippet(client, topic)
    
    if topic not in st.session_state.visual_snippets:
        st.session_state.visual_snippets[topic] = []
    
    st.session_state.visual_snippets[topic].append(next_snippet)
    return st.session_state.visual_snippets[topic]

# ==========================================
# Checklist Agent (RAG-Enhanced)
# ==========================================
def build_tax_checklist(client: OpenAI, chat_messages, user_profile: dict):
    """Build checklist using conversation and RAG context"""
    if not chat_messages:
        return st.session_state.checklist
    
    convo_lines = []
    for m in chat_messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        convo_lines.append(f"{role.upper()}: {content}")
    convo_text = "\n\n".join(convo_lines)
    
    # Get RAG context for better checklist generation
    rag_context = ""
    profile_type = user_profile.get("employment_status", "")
    visa_status = user_profile.get("on_visa", "")
    
    if "Student" in profile_type and visa_status == "Yes":
        rag_context = rag_search("Form 1040-NR international student F-1 requirements", k=2)
    elif "professional" in profile_type.lower():
        rag_context = rag_search("Form 1040 W-2 filing requirements", k=2)

    system_prompt = f"""You are the CHECKLIST AGENT for a US tax-filing assistant.

Your job is to maintain a hierarchical checklist of tax filing tasks based on:
1) The conversation so far
2) The user's profile
3) IRS documentation context

{"Relevant IRS Context:" + chr(10) + rag_context[:500] if rag_context else ""}

You MUST return ONLY valid JSON in this EXACT format:
{{
  "sections": [
    {{
      "heading": "Collect W-2 forms",
      "status": "pending",
      "details": [
        {{"item": "Collect W-2 from each employer", "status": "pending"}},
        {{"item": "Record wages (Box 1)", "status": "pending"}}
      ]
    }}
  ]
}}

Rules:
- Use ACTION headings (e.g., "Collect W-2 forms", "Complete Form 1040-NR")
- Include 3-10 detailed sub-items per section with specific box numbers
- Mark "done" ONLY if user explicitly mentioned completing it
- Tailor to profile:
  * Students/F-1: Form 1098-T, scholarship income, on-campus W-2s, Form 1040-NR
  * Working professionals: W-2, 1099, Form 1040
  * Self-employed: 1099-NEC/1099-K, Schedule C, estimated taxes
- Provide 4-10 sections total
- Return ONLY JSON, no explanation text"""

    user_profile_text = json.dumps(user_profile, indent=2)

    try:
        checklist_resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User profile:\n{user_profile_text}"},
                {"role": "user", "content": f"Conversation so far:\n\n{convo_text}"},
            ],
        )

        raw = checklist_resp.choices[0].message.content or ""
        
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
            
            # Calculate completion percentage
            done_count = sum(1 for d in normalized_details if d["status"] == "done")
            completion = int((done_count / len(normalized_details)) * 100) if normalized_details else 0
            
            normalized_sections.append({
                "heading": heading,
                "status": "done" if completion == 100 else "pending",
                "completion": completion,
                "details": normalized_details,
            })

        if normalized_sections:
            return normalized_sections
    except Exception as e:
        print(f"Checklist error: {e}")

    return st.session_state.checklist

# ==========================================
# Intake Agent System Prompt (RAG-Enhanced)
# ==========================================
def get_intake_system_prompt(user_profile: dict) -> str:
    """Get intake system prompt with RAG context"""
    base_prompt = """You are the INTAKE AGENT for an AI tax assistant.

Your role:
- Talk to the user in a friendly, structured way
- Ask step-by-step questions to understand their tax situation
- Explain what you're doing and what information you need
- Respond naturally to questions like "I need help with W-2" or "I'm on F-1 visa"
- DO NOT mention any internal checklist or multi-agent architecture

Another hidden agent maintains a detailed checklist based on your conversation."""

    # Add RAG context based on user profile
    rag_context = ""
    if user_profile.get("on_visa") == "Yes":
        rag_context = rag_search("nonresident alien F-1 student tax filing", k=1)
    elif user_profile.get("w2_status") and "self-employed" in user_profile.get("w2_status", "").lower():
        rag_context = rag_search("self-employment tax 1099-NEC Schedule C", k=1)
    
    if rag_context:
        base_prompt += f"\n\nRelevant IRS guidance you can reference:\n{rag_context[:400]}"
    
    return base_prompt

# ==========================================
# OpenAI Setup
# ==========================================
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.warning("⚠️ Set your OpenAI API key in environment variable OPENAI_API_KEY")
    openai_api_key = st.text_input("Enter OpenAI API Key:", type="password")
    if not openai_api_key:
        st.stop()

client = OpenAI(api_key=openai_api_key)

# ==========================================
# Page Title & Legal Disclaimer Banner
# ==========================================
st.title("💬 AI Tax Assistant")
st.caption("Powered by OpenAI GPT + RAG (ChromaDB) with Visual Form Mapping")

# Legal Disclaimer Banner (Always Visible)
st.markdown("""
<div class="disclaimer-banner">
⚠️ <strong>IMPORTANT:</strong> This AI assistant is for <strong>educational purposes only</strong>. 
It is NOT a substitute for professional tax advice from a CPA or tax attorney. 
You are solely responsible for the accuracy of your tax filings.
</div>
""", unsafe_allow_html=True)

# Full disclaimer in expander
with st.expander("📜 Full Legal Disclaimer & Privacy Notice", expanded=False):
    tab1, tab2 = st.tabs(["⚖️ Legal Disclaimer", "🔒 Privacy Notice"])
    
    with tab1:
        st.markdown(LEGAL_DISCLAIMER)
    
    with tab2:
        st.markdown(PRIVACY_NOTICE)
    
    # Acknowledgment checkbox
    if 'disclaimer_acknowledged' not in st.session_state:
        st.session_state.disclaimer_acknowledged = False
    
    acknowledged = st.checkbox(
        "I understand this tool is for educational purposes only and does not constitute professional tax advice.",
        value=st.session_state.disclaimer_acknowledged,
        key="disclaimer_checkbox"
    )
    st.session_state.disclaimer_acknowledged = acknowledged

# Check RAG status
db = load_vector_db()
if db:
    st.success("✅ RAG Database Connected", icon="🗄️")
else:
    st.warning("⚠️ RAG Database not found. Visual guides will use general knowledge only.")

# ==========================================
# Initialize Session Memory
# ==========================================
memory_manager = load_memory_manager()

if 'session_id' not in st.session_state:
    st.session_state.session_id = get_or_create_session_id()

if 'session_loaded' not in st.session_state and memory_manager:
    session = load_session_from_memory(memory_manager, st.session_state.session_id)
    st.session_state.user_session = session
    load_session_state_from_memory(memory_manager, session)
    st.session_state.session_loaded = True

if memory_manager:
    st.success("✅ Session Memory Connected", icon="💾")
else:
    st.info("ℹ️ Session memory not available. Progress won't persist across sessions.")

# ==========================================
# Sidebar
# ==========================================
with st.sidebar:
    # Privacy Notice at top
    with st.expander("🔒 Privacy & Data Notice", expanded=False):
        st.markdown(PRIVACY_NOTICE)
    
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
    
    st.session_state.user_profile = {
        "employment_status": user_type,
        "on_visa": visa_status,
        "w2_status": w2_status,
    }
    
    st.divider()
    
    # ==========================================
    # Document Upload Section
    # ==========================================
    st.subheader("📎 Upload Documents")
    
    # Upload Warning
    st.markdown("""
    <div class="upload-warning">
    ⚠️ <strong>Before uploading:</strong> We recommend using documents with 
    sample/redacted SSNs. Any detected sensitive data will be automatically masked.
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_doc = st.file_uploader(
        "Upload tax document (PDF/DOCX/TXT)",
        type=["pdf", "docx", "txt"],
        key="doc_uploader"
    )
    
    if uploaded_doc:
        with st.spinner("📄 Extracting & securing text..."):
            extracted_text, pii_counts, pii_warning = extract_text_from_file(uploaded_doc)
            st.session_state.uploaded_doc_text = extracted_text
            st.session_state.uploaded_doc_name = uploaded_doc.name
        
        st.success(f"✅ {uploaded_doc.name}")
        
        # Show PII warning if detected
        if pii_counts:
            st.warning(f"🔒 Masked {sum(pii_counts.values())} sensitive item(s)")
            with st.expander("View PII Details", expanded=False):
                st.markdown(pii_warning)
        
        with st.expander("📄 Preview (Masked)", expanded=False):
            st.text_area("Content", extracted_text[:500] + "...", height=100, disabled=True)
        
        if st.button("🗑️ Clear Doc", key="clear_doc"):
            st.session_state.uploaded_doc_text = None
            st.session_state.uploaded_doc_name = None
            st.rerun()
    
    # Image upload (OCR)
    uploaded_img = st.file_uploader(
        "Upload W-2/1099 Image (OCR)",
        type=["png", "jpg", "jpeg"],
        key="img_uploader"
    )
    
    if uploaded_img:
        st.image(uploaded_img, caption="Uploaded", use_container_width=True)
        
        with st.spinner("🔍 Running OCR & securing data..."):
            ocr_text, pii_counts, pii_warning = extract_text_from_image(uploaded_img)
            st.session_state.uploaded_img_text = ocr_text
            st.session_state.uploaded_img_name = uploaded_img.name
        
        st.success("✅ OCR completed")
        
        # Show PII warning if detected
        if pii_counts:
            st.warning(f"🔒 Masked {sum(pii_counts.values())} sensitive item(s)")
            with st.expander("View PII Details", expanded=False):
                st.markdown(pii_warning)
        
        with st.expander("🔍 OCR Result (Masked)", expanded=False):
            st.text_area("Extracted", ocr_text[:500], height=100, disabled=True)
        
        if st.button("🗑️ Clear Image", key="clear_img"):
            st.session_state.uploaded_img_text = None
            st.session_state.uploaded_img_name = None
            st.rerun()
    
    st.divider()
    
    # ==========================================
    # Session Management Section
    # ==========================================
    st.subheader("💾 Session Management")
    
    if 'session_id' in st.session_state:
        st.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")
        
        # Session metrics
        if 'user_session' in st.session_state and st.session_state.user_session:
            session = st.session_state.user_session
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Profile", f"{getattr(session, 'profile_completion', 0):.0f}%")
            with col2:
                st.metric("Checklist", f"{getattr(session, 'checklist_completion', 0):.0f}%")
        
        st.caption(f"🔗 Resume: Add `?session_id={st.session_state.session_id}` to URL")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 Copy ID", key="copy_session", use_container_width=True):
                st.code(st.session_state.session_id, language=None)
        
        with col2:
            if st.button("🗑️ Clear", key="clear_session", use_container_width=True):
                if memory_manager:
                    memory_manager.delete_session(st.session_state.session_id)
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        # Load previous session
        with st.expander("🔄 Load Previous Session", expanded=False):
            load_session_id = st.text_input(
                "Enter Session ID:",
                placeholder="xxxxxxxx-xxxx-...",
                key="load_session_input"
            )
            
            if st.button("Load", key="load_session_btn"):
                if load_session_id and memory_manager:
                    loaded = memory_manager.get_session(load_session_id)
                    if loaded:
                        st.session_state.session_id = load_session_id
                        st.session_state.user_session = loaded
                        st.session_state.session_loaded = False
                        st.success("✅ Session loaded!")
                        st.rerun()
                    else:
                        st.error("❌ Session not found")
    
    st.divider()
    
    # ==========================================
    # Live Checklist
    # ==========================================
    st.header("🧾 Live Tax-filing Checklist")
    
    if st.session_state.checklist:
        # Overall progress
        all_sections = st.session_state.checklist
        total_completion = sum(s.get('completion', 0) for s in all_sections) / len(all_sections)
        st.progress(total_completion / 100)
        st.caption(f"Overall Progress: {total_completion:.0f}%")
        
        with st.expander("Checklist (auto-updated)", expanded=True):
            for section in st.session_state.checklist:
                heading = section.get("heading", "Unnamed section")
                completion = section.get("completion", 0)
                details = section.get("details", [])
                
                status_emoji = "✅" if completion == 100 else "⏳"
                st.markdown(f"{status_emoji} **{heading}** ({completion}%)")
                
                for det in details:
                    d_item = det.get("item", "")
                    d_status = det.get("status", "pending").lower()
                    d_emoji = "✅" if d_status == "done" else "⏳"
                    st.markdown(f"- {d_emoji} {d_item}")
                
                st.markdown("")
    else:
        st.info("💡 Start chatting to see your personalized checklist!")

# ==========================================
# Active Documents Indicator
# ==========================================
if st.session_state.get('uploaded_doc_text') or st.session_state.get('uploaded_img_text'):
    active_docs = []
    if st.session_state.get('uploaded_doc_name'):
        active_docs.append(f"📄 {st.session_state.uploaded_doc_name}")
    if st.session_state.get('uploaded_img_name'):
        active_docs.append(f"🖼️ {st.session_state.uploaded_img_name}")
    
    st.info(f"📎 Active documents: {', '.join(active_docs)}")
    
    if st.button("🗑️ Clear All Documents"):
        st.session_state.uploaded_doc_text = None
        st.session_state.uploaded_img_text = None
        st.session_state.uploaded_doc_name = None
        st.session_state.uploaded_img_name = None
        st.rerun()

# ==========================================
# Main Chat UI
# ==========================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Tell me how I can help with your taxes today...")

if prompt:
    # Check for PII in user input and mask it
    detected_pii = PIIHandler.detect_pii(prompt)
    masked_prompt, pii_counts = PIIHandler.mask_pii(prompt)
    
    # Build context with uploaded documents
    context_parts = []
    
    if st.session_state.get('uploaded_doc_text'):
        context_parts.append(f"[Document: {st.session_state.uploaded_doc_name}]\n{st.session_state.uploaded_doc_text[:2000]}")
    
    if st.session_state.get('uploaded_img_text'):
        context_parts.append(f"[OCR from: {st.session_state.uploaded_img_name}]\n{st.session_state.uploaded_img_text}")
    
    full_prompt = "\n\n".join(context_parts) + f"\n\nUser Question: {masked_prompt}" if context_parts else masked_prompt
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
        if context_parts:
            st.caption("📎 *Using uploaded documents*")
        
        # Show PII warning if detected
        if pii_counts:
            st.markdown(f"""
            <div class="pii-warning">
            🔒 <strong>Privacy Protection:</strong> We detected and masked {sum(pii_counts.values())} 
            sensitive item(s) in your message before processing.
            </div>
            """, unsafe_allow_html=True)
    
    # Get RAG-enhanced system prompt
    system_prompt = get_intake_system_prompt(st.session_state.user_profile)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.status("🤖 Thinking...", expanded=False) as status:
            # Search RAG for relevant context
            rag_results = rag_search(prompt, k=2)
            
            # Build messages with RAG context
            messages = [
                {"role": "system", "content": system_prompt},
                *st.session_state.messages[:-1],  # Previous messages
                {"role": "user", "content": f"{full_prompt}\n\n[Relevant IRS Info: {rag_results[:500]}]" if rag_results else full_prompt}
            ]
            
            status.update(label="✅ Generating response...", state="running")
        
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            stream=True,
            messages=messages,
        )
        response_text = st.write_stream(stream)
        
        # Clear used documents
        if context_parts:
            st.caption("📎 *Documents processed*")
            st.session_state.uploaded_doc_text = None
            st.session_state.uploaded_img_text = None
            st.session_state.uploaded_doc_name = None
            st.session_state.uploaded_img_name = None
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    # Sync to persistent memory
    if memory_manager and 'user_session' in st.session_state:
        sync_session_to_memory(
            memory_manager,
            st.session_state.user_session,
            st.session_state.user_profile,
            st.session_state.checklist,
            st.session_state.messages
        )
    
    # Update checklist
    with st.status("📋 Updating checklist...", expanded=False):
        st.session_state.checklist = build_tax_checklist(
            client,
            st.session_state.messages,
            st.session_state.user_profile,
        )

# ==========================================
# Visual Help Section (RAG-Enhanced)
# ==========================================
st.divider()
st.subheader("🧾 Visual Form Mapping Guide (RAG-Enhanced)")

st.markdown("""
Click **"Show Next Step"** to see step-by-step visual guides for mapping your tax forms.
The system uses IRS documentation from the RAG database for accurate mappings.
""")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("🧾 Show Next Step", key="visual_next", use_container_width=True):
        with st.spinner("🔍 Generating with RAG..."):
            if not st.session_state.current_visual_topic:
                st.session_state.current_visual_topic = infer_visual_topic(client)
            
            topic = st.session_state.current_visual_topic
            get_next_visual_snippets(client, topic)

with col2:
    if st.button("🔄 Reset Visuals", key="visual_reset", use_container_width=True):
        st.session_state.visual_snippets = {}
        st.session_state.current_visual_topic = None
        st.rerun()

with col3:
    topic_options = [
        "Auto-detect",
        "w2_to_1040nr", "w2_to_1040",
        "1098t_to_1040nr", "1098t_to_1040",
        "1099int_to_1040", "1099nec_to_schedule_c"
    ]
    
    selected = st.selectbox("Select Form Mapping:", topic_options, key="topic_select")
    
    if selected != "Auto-detect":
        st.session_state.current_visual_topic = selected

# Display current topic
current_topic = st.session_state.current_visual_topic
if current_topic:
    st.caption(f"📌 Current topic: `{current_topic}`")

# Display all generated snippets
if current_topic:
    snippets = st.session_state.visual_snippets.get(current_topic, [])
    
    if snippets:
        st.markdown("### 📋 Generated Visual Steps")
        
        for i, snippet in enumerate(snippets, 1):
            with st.expander(f"Step {i}", expanded=(i == len(snippets))):
                st.code(snippet, language="markdown")
        
        st.caption(f"*{len(snippets)} step(s) generated. Click 'Show Next Step' for more.*")
    else:
        st.info("👆 Click 'Show Next Step' to generate the first visual guide.")
else:
    st.info("💡 Start a conversation, then click 'Show Next Step' to see form mapping visuals.")

# ==========================================
# IRS Document Search Section
# ==========================================
st.divider()
with st.expander("📚 Search IRS Documents (RAG)", expanded=False):
    st.markdown("Search the IRS document database directly for specific information.")
    
    col_s1, col_s2 = st.columns([3, 1])
    
    with col_s1:
        search_query = st.text_input(
            "Search query:",
            placeholder="e.g., W-2 Box 2 federal withholding",
            key="irs_search"
        )
    
    with col_s2:
        if st.button("🔍 Search", key="search_btn", use_container_width=True):
            if search_query:
                with st.spinner("Searching..."):
                    st.session_state.search_result = rag_search(search_query, k=3)
    
    if st.session_state.get('search_result'):
        st.markdown("### 📄 Search Results")
        st.markdown(st.session_state.search_result)
        
        if st.button("🗑️ Clear Results", key="clear_search"):
            st.session_state.search_result = None
            st.rerun()