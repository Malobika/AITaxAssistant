import streamlit as st
from openai import OpenAI
import streamlit as st
import pandas as pd
import os
from glob import glob

DATA_DIR = "./federal_tax_documents"
PERSIST_FILE = "./saved_table.csv"

# --- LOAD DATA FROM FOLDER ---
def load_all_tables(path):
    all_files = glob(os.path.join(path, "*.csv"))
    tables = []

    for file in all_files:
        df = pd.read_csv(file)
        df["source_file"] = os.path.basename(file)  # optional
        tables.append(df)

    # combine all into one dataframe
    if tables:
        return pd.concat(tables, ignore_index=True)
    return None


# --- SAVE TO PERSISTENT STORAGE ---
def save_persistent(df):
    df.to_csv(PERSIST_FILE, index=False)


# --- MAIN APP ---
st.title("Federal Tax Documents Table Loader")

# If persistent file exists, load that
if os.path.exists(PERSIST_FILE):
    st.success("Loaded table from persistent storage.")
    table = pd.read_csv(PERSIST_FILE)
else:
    # Load fresh from directory
    table = load_all_tables(DATA_DIR)

    if table is not None:
        save_persistent(table)
        st.success("Loaded and saved table from directory.")

if table is not None:
    st.write("📄 **Complete Combined Table:**")
    st.dataframe(table)
else:
    st.warning("No CSV files found in ./federal_tax_documents/")


