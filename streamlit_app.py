import streamlit as st
from openai import OpenAI
# app.py
import os
from glob import glob


import pandas as pd
import tabula  # for PDF -> tables

DATA_DIR = "./federal_tax_documents"
PERSIST_FILE = "./federal_tax_tables.csv"


@st.cache_data
def extract_tables_from_pdf(pdf_path: str) -> pd.DataFrame | None:
    """
    Extract all tables from a single PDF and return as one DataFrame.
    """
    # multiple_tables=True returns a list of DataFrames (one per table found)
    tables = tabula.read_pdf(
        pdf_path,
        pages="all",
        multiple_tables=True
    )

    if not tables:
        return None

    df = pd.concat(tables, ignore_index=True)
    df["source_file"] = os.path.basename(pdf_path)
    return df


@st.cache_data
def build_master_table(data_dir: str) -> pd.DataFrame | None:
    """
    Go through all PDFs in the folder, extract tables, and combine into one big table.
    """
    pdf_files = glob(os.path.join(data_dir, "*.pdf"))
    all_tables = []

    for pdf in pdf_files:
        df = extract_tables_from_pdf(pdf)
        if df is not None:
            all_tables.append(df)

    if not all_tables:
        return None

    master = pd.concat(all_tables, ignore_index=True)
    return master


# app.py (continued)

st.title("Federal Tax PDF → Persistent Table")

table = None

# 1️⃣ If a persisted CSV already exists, load that first
if os.path.exists(PERSIST_FILE):
    table = pd.read_csv(PERSIST_FILE)
    st.success("Loaded table from persistent CSV on disk.")
else:
    # 2️⃣ Otherwise, build from PDFs and then persist
    table = build_master_table(DATA_DIR)

    if table is not None:
        table.to_csv(PERSIST_FILE, index=False)
        st.success("Extracted tables from PDFs and saved to persistent CSV.")
    else:
        st.warning("No tables found in PDFs in ./federal_tax_documents")

# 3️⃣ Show the table if we have it
if table is not None:
    st.subheader("Combined Table from All PDFs")
    st.dataframe(table)

    st.write("Columns detected:")
    st.write(list(table.columns))
