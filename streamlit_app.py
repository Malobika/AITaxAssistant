import streamlit as st
from openai import OpenAI


import pandas as pd
import pdfplumber
import os
from glob import glob

DATA_DIR = "./federal_tax_documents"
PERSIST_FILE = "./federal_tax_tables.csv"


@st.cache_data
def extract_tables_from_pdf(pdf_path: str) -> pd.DataFrame | None:
    """
    Extract all tables from a single PDF into one DataFrame.
    Columns are named col_0, col_1, ... based on position.
    """
    rows = []
    basename = os.path.basename(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            if not tables:
                continue

            for t_idx, table in enumerate(tables):
                for row_idx, row in enumerate(table):
                    if row is None:
                        continue

                    # Build a dict: source info + each column
                    row_dict = {
                        "source_file": basename,
                        "page": page_num,
                        "table_index": t_idx,
                        "row_index": row_idx,
                    }
                    for col_idx, cell in enumerate(row):
                        row_dict[f"col_{col_idx}"] = cell
                    rows.append(row_dict)

    if not rows:
        return None

    return pd.DataFrame(rows)


@st.cache_data
def build_master_table(data_dir: str) -> pd.DataFrame | None:
    """
    Loop over all PDFs in the folder, extract tables, and combine them.
    """
    pdf_files = glob(os.path.join(data_dir, "*.pdf"))
    all_tables = []

    for pdf in pdf_files:
        df = extract_tables_from_pdf(pdf)
        if df is not None and not df.empty:
            all_tables.append(df)

    if not all_tables:
        return None

    master_df = pd.concat(all_tables, ignore_index=True)
    return master_df


st.title("🧾 AI Tax Assistant")

table = None

# 1️⃣ Try to load persisted CSV first
if os.path.exists(PERSIST_FILE):
    table = pd.read_csv(PERSIST_FILE)
    st.success("Loaded table from persistent CSV file.")
else:
    # 2️⃣ Otherwise, build from PDFs and persist
    table = build_master_table(DATA_DIR)
    if table is not None:
        table.to_csv(PERSIST_FILE, index=False)
        st.success("Extracted tables from PDFs and saved to CSV (persistent).")
    else:
        st.warning("No tables found in PDFs under ./federal_tax_documents")

# 3️⃣ Display the table
if table is not None:
    st.subheader("Combined Table from All PDFs")
    st.dataframe(table)

    st.write("Columns in the table:")
    st.write(list(table.columns))


if st.button("Clear All Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("All cache cleared!")