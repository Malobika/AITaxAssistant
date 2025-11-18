__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import glob
from datetime import datetime
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm  # For progress bars

# --- 1. Define Your Data Paths ---
print("=" * 50)
print("Starting Federal Tax Document Processing Pipeline")
print("=" * 50)

# Find all .docx files in the federal_tax_documents folder
doc_files = glob.glob("federal_tax_documents/**/*.docx", recursive=True)

print(f"\n📁 Found {len(doc_files)} .docx files")
print("\nBreakdown by folder:")
faqs_count = len([f for f in doc_files if "FAQs" in f])
forms_count = len([f for f in doc_files if "federal_forms" in f])
instructions_count = len([f for f in doc_files if "federal_instructions" in f])
print(f"  - FAQs: {faqs_count} files")
print(f"  - Forms: {forms_count} files")
print(f"  - Instructions: {instructions_count} files")

# --- 2. Load Documents and Add Metadata ---
print("\n📄 Loading documents and adding metadata...")
all_docs_with_metadata = []
failed_files = []

for doc_path in tqdm(doc_files, desc="Processing files"):
    try:
        # Load a single document
        loader = UnstructuredWordDocumentLoader(doc_path)
        documents = loader.load()
        
        # Extract filename and folder information
        filename = os.path.basename(doc_path)
        folder_path = os.path.dirname(doc_path)
        
        # Determine document type based on folder structure
        doc_type = "unknown"
        if "FAQs" in folder_path:
            doc_type = "faq"
        elif "federal_forms" in folder_path:
            doc_type = "form"
        elif "federal_instructions" in folder_path:
            doc_type = "instruction"
        
        # Extract form number from filename
        # Handle different filename patterns
        form_number = ""
        base_name = filename.replace('.docx', '').replace('.doc', '')
        
        if base_name.startswith('f'):
            # Form files like f1040.docx -> 1040
            form_number = base_name[1:]
        elif base_name.startswith('i'):
            # Instruction files like i1040gi_instructions.docx -> 1040gi
            form_number = base_name[1:].split('_')[0]
        elif base_name.startswith('fw'):
            # Files like fw2.docx -> w2
            form_number = base_name[1:]
        elif base_name.startswith('p'):
            # Publication files like p15t.docx -> 15t
            form_number = base_name[1:]
        else:
            # For other files, use the base name
            form_number = base_name
        
        # Clean up form number (remove any remaining special characters)
        form_number = form_number.replace('_instructions', '').replace('_', '-')
        
        # Add comprehensive metadata to each document
        for doc in documents:
            doc.metadata.update({
                "source_file": filename,
                "full_path": doc_path,
                "doc_type": doc_type,
                "jurisdiction": "federal",
                "form_number": form_number.upper(),  # Standardize to uppercase
                "year": "2025",  # You can modify this based on your documents
                "processed_date": datetime.now().isoformat(),
                "char_count": len(doc.page_content),
                "folder": folder_path.split('/')[-1] if '/' in folder_path else folder_path.split('\\')[-1]
            })
            
            # Add specific metadata based on document type
            if doc_type == "instruction":
                doc.metadata["is_instruction"] = True
                doc.metadata["instruction_for_form"] = form_number.upper()
            elif doc_type == "form":
                doc.metadata["is_form"] = True
                doc.metadata["form_id"] = form_number.upper()
            elif doc_type == "faq":
                doc.metadata["is_faq"] = True
                doc.metadata["faq_category"] = "general" if "FAQs.docx" in filename else form_number.upper()
        
        all_docs_with_metadata.extend(documents)
        
    except Exception as e:
        print(f"\n⚠️  Error processing {doc_path}: {str(e)}")
        failed_files.append(doc_path)
        continue

print(f"\n✅ Successfully loaded {len(all_docs_with_metadata)} documents")
if failed_files:
    print(f"❌ Failed to load {len(failed_files)} files:")
    for f in failed_files:
        print(f"   - {f}")

# --- 3. Chunking ---
print("\n✂️  Chunking documents...")
print("   Configuration:")
print("   - Chunk size: 1000 characters")
print("   - Chunk overlap: 200 characters")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=9000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
    keep_separator=True
)

all_chunks = text_splitter.split_documents(all_docs_with_metadata)

# Add chunk-specific metadata
for i, chunk in enumerate(all_chunks):
    chunk.metadata["chunk_id"] = i
    chunk.metadata["chunk_size"] = len(chunk.page_content)

print(f"✅ Created {len(all_chunks)} chunks from {len(all_docs_with_metadata)} documents")
print(f"   Average chunks per document: {len(all_chunks) / len(all_docs_with_metadata):.1f}")

# --- 4. Initialize Embeddings ---
print("\n🤖 Initializing embedding model...")
print("   Model: all-MiniLM-L6-v2")
print("   This may take a minute on first run...")

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # Change to 'cuda' if you have GPU
    encode_kwargs={'normalize_embeddings': True}
)

# --- 5. Create and Save ChromaDB ---
print("\n💾 Building ChromaDB vector database...")
db_directory = "federal_tax_vector_db"

# Delete existing database if it exists (optional - comment out if you want to append)
if os.path.exists(db_directory):
    print(f"   ⚠️  Existing database found at '{db_directory}'. It will be replaced.")
    import shutil
    shutil.rmtree(db_directory)

# Create the database
db = Chroma.from_documents(
    documents=all_chunks,
    embedding=embedding_model,
    persist_directory=db_directory,
    collection_name="federal_tax_documents",
    collection_metadata={"description": "Federal tax forms, instructions, and FAQs"}
)

# --- 6. Test the Database ---
print("\n🔍 Testing the database with a sample query...")
test_query = "What is Form 1040?"
results = db.similarity_search(test_query, k=3)

print(f"\nQuery: '{test_query}'")
print(f"Top 3 results:")
for i, result in enumerate(results, 1):
    print(f"\n  Result {i}:")
    print(f"    Source: {result.metadata.get('source_file', 'Unknown')}")
    print(f"    Type: {result.metadata.get('doc_type', 'Unknown')}")
    print(f"    Form: {result.metadata.get('form_number', 'N/A')}")
    print(f"    Preview: {result.page_content[:100]}...")

# --- 7. Summary ---
print("\n" + "=" * 50)
print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 50)
print(f"\n📊 Final Statistics:")
print(f"   - Documents processed: {len(all_docs_with_metadata)}")
print(f"   - Total chunks created: {len(all_chunks)}")
print(f"   - Database saved to: '{db_directory}/'")
print(f"   - Collection name: 'federal_tax_documents'")

