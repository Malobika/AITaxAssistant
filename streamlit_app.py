import streamlit as st
from openai import OpenAI

import streamlit as st
import pandas as pd
import os
from docx import Document  # from python-docx


DOC_PATH = "./federal_tax_documents/federal_forms/f1040.docx"  # change to your file


def extract_lines_from_docx(path: str):
    """Extract non-empty text lines from a .docx file."""
    doc = docx.Document(path)
    lines = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            lines.append(text)
    return lines


def lines_to_table(lines):
    """
    Convert lines like 'Field: Value' into a 2-column table.
    Lines without ':' go into 'value' only.
    """
    rows = []
    for line in lines:
        if ":" in line:
            field, value = line.split(":", 1)
            rows.append({
                "field": field.strip(),
                "value": value.strip()
            })
        else:
            rows.append({
                "field": "",
                "value": line.strip()
            })
    return pd.DataFrame(rows)


st.title("🧾 ICON... AI Tax Assistant – DOC Extractor")

if not os.path.exists(DOC_PATH):
    st.error(f"File not found: {DOC_PATH}")
else:
    # 1️⃣ Extract text lines
    lines = extract_lines_from_docx(DOC_PATH)

    st.subheader("Raw Text (Line by Line)")
    st.write(lines)  # or st.text("\n".join(lines))

    # 2️⃣ Convert to table
    df = lines_to_table(lines)

    st.subheader("Converted Table")
    st.dataframe(df)

    st.write("Columns:", list(df.columns))
