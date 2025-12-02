import streamlit as st
from openai import OpenAI

import pandas as pd
import os
from docx import Document  # from python-docx
import sqlite3
from glob import glob
import pdfplumber
import easyocr
from PIL import Image
import numpy as np
import json

# ---------------------------------------------------------
# 🌐 Streamlit config & session state
# ---------------------------------------------------------
st.set_page_config(page_title="AI Tax Assistant", page_icon="💬")

if "messages" not in st.session_state:
    st.session_state.messages = []

# checklist structure:
# [
#   {
#     "heading": "Collect W-2 forms",
#     "status": "pending" | "done",
#     "details": [
#        {"item": "...", "status": "pending" | "done"},
#        ...
#     ]
#   },
#   ...
# ]
if "checklist" not in st.session_state:
    st.session_state.checklist = []

if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------
DATA_DIR = "./federal_tax_documents/federal_forms"
DB_PATH = "./documents.db"


# ---------------------------------------------------------
# ---------- DB SETUP ----------
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            file_path TEXT,
            file_type TEXT
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS extracted_rows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER,
            row_index INTEGER,
            col_index INTEGER,
            text TEXT,
            FOREIGN KEY(doc_id) REFERENCES documents(id)
        );
        """
    )

    conn.commit()
    conn.close()


def upsert_document(filename: str, file_path: str, file_type: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO documents (filename, file_path, file_type)
        VALUES (?, ?, ?)
        ON CONFLICT(filename) DO UPDATE SET file_path=excluded.file_path,
                                            file_type=excluded.file_type;
        """,
        (filename, file_path, file_type),
    )
    conn.commit()

    # get id
    cur.execute("SELECT id FROM documents WHERE filename = ?", (filename,))
    doc_id = cur.fetchone()[0]

    conn.close()
    return doc_id


def clear_extracted_for_doc(doc_id: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM extracted_rows WHERE doc_id = ?", (doc_id,))
    conn.commit()
    conn.close()


def insert_extracted_rows(doc_id: int, rows: list[dict]):
    """
    rows: list of dicts {row_index, col_index, text}
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT INTO extracted_rows (doc_id, row_index, col_index, text)
        VALUES (?, ?, ?, ?)
        """,
        [(doc_id, r["row_index"], r["col_index"], r["text"]) for r in rows],
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------
# ---------- EXTRACTION HELPERS ----------
# ---------------------------------------------------------
def extract_pdf_tables(file_path: str) -> list[dict]:
    """
    Extract all tables from a PDF as list of {row_index, col_index, text}.
    """
    extracted = []
    row_counter = 0

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if not tables:
                continue
            for table in tables:
                for row in table:
                    if row is None:
                        continue
                    for col_idx, cell in enumerate(row):
                        text = (cell or "").strip()
                        extracted.append(
                            {
                                "row_index": row_counter,
                                "col_index": col_idx,
                                "text": text,
                            }
                        )
                    row_counter += 1

    return extracted


def extract_docx_lines(file_path: str) -> list[dict]:
    """
    Extract paragraphs from a DOCX as a single 'column' table:
    col_0 = text
    """
    extracted = []
    doc = Document(file_path)

    for idx, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if not text:
            continue
        extracted.append(
            {
                "row_index": idx,
                "col_index": 0,
                "text": text,
            }
        )

    return extracted


# ---------------------------------------------------------
# ---------- PROCESS ALL DOCUMENTS ----------
# ---------------------------------------------------------
def process_all_documents():
    """
    Scan DATA_DIR for .pdf and .docx,
    extract content, and store in SQLite.
    """
    pdf_files = glob(os.path.join(DATA_DIR, "*.pdf"))
    docx_files = glob(os.path.join(DATA_DIR, "*.docx"))

    total_docs = 0
    total_rows = 0

    for file_path in pdf_files + docx_files:
        filename = os.path.basename(file_path)
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            file_type = "pdf"
            rows = extract_pdf_tables(file_path)
        elif ext == ".docx":
            file_type = "docx"
            rows = extract_docx_lines(file_path)
        else:
            continue  # skip unknown

        doc_id = upsert_document(filename, file_path, file_type)
        clear_extracted_for_doc(doc_id)

        if rows:
            insert_extracted_rows(doc_id, rows)
            total_rows += len(rows)

        total_docs += 1

    return total_docs, total_rows


def get_documents_df() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT id, filename, file_type FROM documents", conn)
    conn.close()
    return df


def get_extracted_for_doc(doc_id: int) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT row_index, col_index, text
        FROM extracted_rows
        WHERE doc_id = ?
        ORDER BY row_index, col_index
        """,
        conn,
        params=(doc_id,),
    )
    conn.close()
    return df


# ---------------------------------------------------------
# OCR SETUP
# ---------------------------------------------------------
@st.cache_resource
def load_ocr():
    return easyocr.Reader(["en"], gpu=False)


ocr_reader = load_ocr()


# ---------------------------------------------------------
# LLM-based dynamic checklist helper
# ---------------------------------------------------------
def build_tax_checklist(client: OpenAI, chat_messages, user_profile: dict):
    """
    Use the LLM to generate/update a tax-filing checklist from the chat history
    and the user's profile.

    Expected JSON shape:

    {
      "sections": [
        {
          "heading": "Collect W-2 forms",
          "status": "pending",
          "details": [
            {"item": "Collect W-2 from each employer", "status": "pending"},
            {"item": "Confirm employer name, address, and EIN (Box b)", "status": "pending"},
            {"item": "Record wages (Box 1) and federal income tax withheld (Box 2)", "status": "pending"}
          ]
        },
        ...
      ]
    }
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
You are a careful US tax-filing assistant.

Your job is to maintain a hierarchical checklist of tax filing tasks and
information the user needs, based on:
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
- Use clear ACTION headings that reference specific forms when relevant, e.g.:
  * "Collect W-2 forms"
  * "Gather 1099-INT and 1099-DIV statements"
  * "Gather 1099-NEC / 1099-K for self-employment income"
  * "Collect Form 1098-T and tuition payment records"
  * "Confirm filing information on Form 1040"
- Under each heading, include 3–10 detailed sub-items ("details") that describe
  concrete information to collect or verify, such as:
  * Box numbers on W-2, 1099-INT, 1099-DIV, 1099-NEC, 1099-K
  * Employer name/EIN, payer details
  * Gross amounts, tax withheld, dates, etc.
- Mark a detail as "done" ONLY if the conversation clearly indicates the user
  has already provided that information or completed that step. Otherwise "pending".
- The section "status" should be "done" ONLY if all or nearly all of its details
  appear to be completed from the conversation.
- Tailor sections to the profile:
  * Students / on F-1: likely Form 1098-T, scholarship income, campus employment W-2s.
  * Working professionals: W-2, 1099 income, retirement contributions, etc.
  * Self-employed / 1099: business income, expenses, estimated taxes.
- Provide between 4 and 10 sections total.
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
# ---------- STREAMLIT APP ----------
# ---------------------------------------------------------

st.title("💬 AI Assistant Chatbot")
st.write(
    "This is a simple chatbot that uses OpenAI's GPT model to help fill your tax forms."
)

# init DB on startup
init_db()

docs, rows = process_all_documents()
st.success(f"Processed {docs} document(s), extracted {rows} cell(s).")

# List all documents from DB
if os.path.exists(DB_PATH):
    docs_df = get_documents_df()
else:
    docs_df = pd.DataFrame(columns=["id", "filename", "file_type"])

st.subheader("Documents in persistent DB")

if docs_df.empty:
    st.info("No documents indexed yet.")
else:
    st.dataframe(docs_df)

    # Select a document to view
    doc_options = {
        f'{row["filename"]} ({row["file_type"]})': row["id"]
        for _, row in docs_df.iterrows()
    }
    selected_label = st.selectbox(
        "Select a document to view extracted data:", list(doc_options.keys())
    )
    selected_doc_id = doc_options[selected_label]

    # Show extracted rows for that document
    extracted_df = get_extracted_for_doc(selected_doc_id)

    st.subheader(f"Extracted table for: {selected_label}")
    if extracted_df.empty:
        st.warning("No extracted data for this document.")
    else:
        st.dataframe(extracted_df)

    # Download button for the original document
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT file_path FROM documents WHERE id = ?", (selected_doc_id,))
    file_path = cur.fetchone()[0]
    conn.close()

    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            st.download_button(
                label="Download original document",
                data=f,
                file_name=os.path.basename(file_path),
                mime="application/octet-stream",
            )
    else:
        st.error(f"File not found on disk: {file_path}")

# ---------------------------------------------------------
# 📄 / 🖼️ Upload Area: 2 Columns (Documents & Images)
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Upload a document")
    doc_file = st.file_uploader(
        "Upload a PDF / TXT / DOCX",
        type=["pdf", "txt", "docx"],
        key="doc_uploader",
    )

with col2:
    st.subheader("🖼️ Upload an image")
    img_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg"],
        key="img_uploader",
    )

uploaded_text = None  # text extracted from uploaded document (if any)

# ---------------------------------------------------------
# 📄 Handle document upload + preview
# ---------------------------------------------------------
try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

if doc_file is not None:
    st.markdown(f"**Document uploaded:** `{doc_file.name}`")

    if doc_file.type == "text/plain":
        # Simple text file
        uploaded_text = doc_file.read().decode("utf-8", errors="ignore")

    elif doc_file.type == "application/pdf":
        if PdfReader is None:
            st.warning("PyPDF2 is not installed. Run `pip install pypdf2` to read PDFs.")
        else:
            reader = PdfReader(doc_file)
            text_chunks = []
            for page in reader.pages[:5]:  # limit pages to avoid huge output
                text_chunks.append(page.extract_text() or "")
            uploaded_text = "\n".join(text_chunks)

    elif doc_file.type in [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]:
        if DocxDocument is None:
            st.warning("python-docx is not installed. Run `pip install python-docx`.")
        else:
            doc = DocxDocument(doc_file)
            paragraphs = [p.text for p in doc.paragraphs]
            uploaded_text = "\n".join(paragraphs)

    # Show the extracted / raw text (if any)
    if uploaded_text:
        with st.expander("📄 Preview extracted document text"):
            st.text(uploaded_text)
    else:
        st.info("Document uploaded but no text could be extracted.")

# ---------------------------------------------------------
# 🖼️ Handle image upload + OCR
# ---------------------------------------------------------
if img_file is not None:
    st.image(img_file, caption="Uploaded image", use_container_width=True)

    # Convert to NumPy image
    image = Image.open(img_file)
    image_np = np.array(image)

    st.write("🔍 Extracting text...")

    # Run OCR
    results = ocr_reader.readtext(image_np)

    extracted_text = "\n".join([res[1] for res in results])

    st.subheader("📄 OCR Output")
    st.text(extracted_text)

# ---------------------------------------------------------
# 🧍 User profile (sidebar) — student/working, visa, W2
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

    # Save profile into session_state
    st.session_state.user_profile = {
        "employment_status": user_type,
        "on_visa": visa_status,
        "w2_status": w2_status,
    }

    if st.button("Reset checklist"):
        st.session_state.checklist = []

# ---------------------------------------------------------
# 🔑 OpenAI setup & Chat UI
# ---------------------------------------------------------
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.info("Update your key in the environment variable OPENAI_API_KEY.", icon="🗝️")
else:
    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input("Do you want to file your taxes?")
    if prompt:
        # Optionally inject uploaded document text into the context
        if uploaded_text:
            prompt_with_context = (
                "The user has uploaded the following document text:\n\n"
                f"{uploaded_text}\n\n"
                f"User question: {prompt}"
            )
        else:
            prompt_with_context = prompt

        # Store and display user's message
        st.session_state.messages.append(
            {"role": "user", "content": prompt_with_context}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response (streaming)
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )

        # Stream response and save it
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

        # 🔄 Update checklist after each turn, based on profile + conversation
        st.session_state.checklist = build_tax_checklist(
            client,
            st.session_state.messages,
            st.session_state.user_profile,
        )

# ---------------------------------------------------------
# 🧾 Checklist display in sidebar (with headings + detailed sub-items)
# ---------------------------------------------------------
with st.sidebar:
    st.header("🧾 Tax-filing Checklist")

    if st.session_state.checklist:
        with st.expander("View / update checklist", expanded=True):
            for s_idx, section in enumerate(st.session_state.checklist):
                heading = section.get("heading", "Unnamed section")
                sec_status = section.get("status", "pending").lower()
                details = section.get("details", [])

                st.markdown(f"**{heading}**")

                # Section-level checkbox
                section_done_default = sec_status == "done"
                new_section_done = st.checkbox(
                    f"Mark section '{heading}' as complete",
                    value=section_done_default,
                    key=f"section_done_{s_idx}",
                )

                # If user marks section complete, mark all details as done as well
                if new_section_done:
                    st.session_state.checklist[s_idx]["status"] = "done"
                else:
                    st.session_state.checklist[s_idx]["status"] = "pending"

                # Sub-items (details) with slight "indentation" in label
                for d_idx, det in enumerate(details):
                    d_label = det.get("item", "")
                    d_status = det.get("status", "pending").lower()
                    d_checked_default = d_status == "done"

                    new_d_checked = st.checkbox(
                        "• " + d_label,
                        value=d_checked_default,
                        key=f"detail_{s_idx}_{d_idx}",
                    )
                    st.session_state.checklist[s_idx]["details"][d_idx]["status"] = (
                        "done" if new_d_checked else "pending"
                    )

                st.markdown("---")
    else:
        st.info("Start chatting to generate a personalized tax-filing checklist.")
