# Fix sqlite3 issue (must be at the very top)
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except (ImportError, KeyError):
    pass

import streamlit as st
import os
import sys

# Ensure we can find tax_brain_merged
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from tax_brain_merged import TaxOrchestrator, UserProfile, PIIHandler, LEGAL_DISCLAIMER, PRIVACY_NOTICE
from sessionmemory import SessionMemoryManager, UserSession, get_memory_manager

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
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="AI Tax Assistant",
    page_icon="📋",
    layout="wide"
)

# ==========================================
# Custom CSS for Disclaimer Banner
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
# OCR Model Loading (Cached)
# ==========================================
@st.cache_resource
def load_ocr():
    if OCR_AVAILABLE:
        return easyocr.Reader(["en"], gpu=False)
    return None

# ==========================================
# Document Text Extraction (with PII masking)
# ==========================================
def extract_text_from_file(uploaded_file):
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
    
    # Mask PII in extracted text
    masked_text, pii_counts = PIIHandler.mask_pii(text)
    warning = PIIHandler.get_pii_warning(
        {k: [f"[{v} instances]"] for k, v in pii_counts.items()}
    ) if pii_counts else ""
    
    return masked_text, pii_counts, warning

def extract_text_from_image(uploaded_image):
    if not OCR_AVAILABLE:
        return "❌ OCR not installed.", {}, ""
    
    try:
        ocr_reader = load_ocr()
        image = Image.open(uploaded_image)
        image_np = np.array(image)
        results = ocr_reader.readtext(image_np)
        text = "\n".join([res[1] for res in results])
        
        # Mask PII in OCR text
        masked_text, pii_counts = PIIHandler.mask_pii(text)
        warning = PIIHandler.get_pii_warning(
            {k: [f"[{v} instances]"] for k, v in pii_counts.items()}
        ) if pii_counts else ""
        
        return masked_text, pii_counts, warning
    except Exception as e:
        return f"❌ OCR Error: {str(e)}", {}, ""

# ==========================================
# Sidebar Rendering
# ==========================================
def render_sidebar():
    with st.sidebar:
        # Privacy Notice at top of sidebar
        with st.expander("🔒 Privacy & Data Notice", expanded=False):
            st.markdown(PRIVACY_NOTICE)
        
        st.header("📋 User Profile")
        
        user_profile = st.session_state.get('user_profile', UserProfile())
        
        # Profile completion
        try:
            completeness = st.session_state.orchestrator.intake_agent.check_completeness(user_profile)
            completion_rate = completeness['completion_rate']
            
            st.progress(completion_rate / 100)
            st.caption(f"Profile Completion: {completion_rate:.0f}%")
            
            st.divider()
            
            profile_fields = {
                "🌐 Citizenship": user_profile.citizenship_status,
                "🎓 Student Status": user_profile.student_status,
                "💼 Employment": user_profile.employment_details,
                "💰 Income": f"${user_profile.income:,}" if user_profile.income else None,
                "📍 State": user_profile.residency_state,
            }
            
            for label, value in profile_fields.items():
                st.markdown(f"**{label}:** {value or '⬜ Not provided'}")
        except Exception as e:
            st.error(f"Error loading profile: {e}")
        
        st.divider()
        
        # Document upload
        st.subheader("📎 Upload Documents")
        
        uploaded_doc = st.file_uploader(
            "Upload tax document (PDF/DOCX/TXT)",
            type=["pdf", "docx", "txt"],
            key="doc_uploader"
        )
        
        if uploaded_doc:
            with st.spinner("📄 Extracting text..."):
                extracted_text = extract_text_from_file(uploaded_doc)
                st.session_state.uploaded_doc_text = extracted_text
                st.session_state.uploaded_doc_name = uploaded_doc.name
            
            st.success(f"✅ Extracted from: {uploaded_doc.name}")
            
            with st.expander("📄 Preview", expanded=False):
                st.text_area("Content", extracted_text[:500] + "...", height=150, disabled=True)
            
            if st.button("🗑️ Clear Document", key="clear_doc"):
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
            
            with st.spinner("🔍 Running OCR..."):
                ocr_text = extract_text_from_image(uploaded_img)
                st.session_state.uploaded_img_text = ocr_text
                st.session_state.uploaded_img_name = uploaded_img.name
            
            st.success(f"✅ OCR completed")
            
            if st.button("🗑️ Clear Image", key="clear_img"):
                st.session_state.uploaded_img_text = None
                st.session_state.uploaded_img_name = None
                st.rerun()
        
        st.divider()
        
        # ==========================================
        # Session Management Section
        # ==========================================
        st.subheader("💾 Session Management")
        
        # Show session ID
        if 'session_id' in st.session_state:
            st.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")
            
            # Session summary
            if 'user_session' in st.session_state:
                session = st.session_state.user_session
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Profile", f"{session.profile_completion:.0f}%")
                with col2:
                    st.metric("Checklist", f"{session.checklist_completion:.0f}%")
            
            # Copy session link
            session_url = f"?session_id={st.session_state.session_id}"
            st.caption(f"🔗 Resume link: Add `{session_url}` to URL")
            
            # Session actions
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📋 Copy ID", key="copy_session_id", use_container_width=True):
                    st.code(st.session_state.session_id, language=None)
            
            with col2:
                if st.button("🗑️ Clear Session", key="clear_session", use_container_width=True):
                    # Delete from memory
                    memory_manager = load_memory_manager()
                    memory_manager.delete_session(st.session_state.session_id)
                    
                    # Clear session state
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    
                    st.rerun()
            
            # Load existing session
            with st.expander("🔄 Load Previous Session", expanded=False):
                load_session_id = st.text_input(
                    "Enter Session ID:",
                    placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                    key="load_session_input"
                )
                
                if st.button("Load Session", key="load_session_btn"):
                    if load_session_id:
                        memory_manager = load_memory_manager()
                        loaded_session = memory_manager.get_session(load_session_id)
                        
                        if loaded_session:
                            # Clear current state
                            st.session_state.session_id = load_session_id
                            st.session_state.user_session = loaded_session
                            st.session_state.session_loaded = False  # Force reload
                            st.success("✅ Session loaded!")
                            st.rerun()
                        else:
                            st.error("❌ Session not found")
        
        st.divider()
        
        # Checklist Display
        st.subheader("📋 Tax Filing Checklist")
        
        if st.session_state.get('checklist'):
            all_sections = st.session_state.checklist
            total_completion = sum(s.get('completion', 0) for s in all_sections) / len(all_sections)
            st.progress(total_completion / 100)
            st.caption(f"Overall Progress: {total_completion:.0f}%")
            
            for section in all_sections:
                heading = section.get("heading", "Unnamed")
                completion = section.get("completion", 0)
                details = section.get("details", [])
                status_emoji = "✅" if completion == 100 else "⏳"
                
                with st.expander(f"{status_emoji} {heading} ({completion}%)", expanded=(0 < completion < 100)):
                    st.progress(completion / 100)
                    for detail in details:
                        d_emoji = "✅" if detail.get("status") == "done" else "⏳"
                        st.markdown(f"{d_emoji} {detail.get('item', '')}")
        else:
            st.info("💡 Start chatting to see your checklist!")

# ==========================================
# Visual Help Section (RAG-Enhanced)
# ==========================================
def render_visual_help():
    st.divider()
    st.subheader("🧾 Visual Form Mapping Guide (RAG-Enhanced)")
    
    st.markdown("""
    Click **"Show Next Step"** to see step-by-step visual guides for mapping your tax forms.
    The system automatically detects which forms you're working with based on your conversation.
    """)
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🧾 Show Next Step", key="visual_next_btn", use_container_width=True):
            with st.spinner("🔍 Generating visual guide with RAG..."):
                # Infer topic if not set
                if not st.session_state.get('current_visual_topic'):
                    topic = st.session_state.orchestrator.infer_visual_topic(
                        st.session_state.messages,
                        st.session_state.user_profile
                    )
                    st.session_state.current_visual_topic = topic
                
                # Generate next snippet
                topic = st.session_state.current_visual_topic
                snippet = st.session_state.orchestrator.generate_visual_step(
                    topic,
                    st.session_state.user_profile
                )
                st.session_state.latest_visual_snippet = snippet
    
    with col2:
        if st.button("🔄 Reset Visuals", key="visual_reset_btn", use_container_width=True):
            st.session_state.orchestrator.reset_visuals()
            st.session_state.current_visual_topic = None
            st.session_state.latest_visual_snippet = None
            st.rerun()
    
    with col3:
        # Topic selector
        topic_options = [
            "Auto-detect",
            "w2_to_1040nr", "w2_to_1040",
            "1098t_to_1040nr", "1098t_to_1040",
            "1099int_to_1040", "1099nec_to_schedule_c"
        ]
        
        selected_topic = st.selectbox(
            "Select Form Mapping:",
            topic_options,
            key="visual_topic_select"
        )
        
        if selected_topic != "Auto-detect":
            st.session_state.current_visual_topic = selected_topic
    
    # Display current topic
    current_topic = st.session_state.get('current_visual_topic')
    if current_topic:
        st.caption(f"📌 Current topic: `{current_topic}`")
    
    # Display all generated snippets
    if current_topic:
        snippets = st.session_state.orchestrator.get_visual_snippets(current_topic)
        
        if snippets:
            st.markdown("### 📋 Generated Visual Steps")
            
            for i, snippet in enumerate(snippets, 1):
                with st.expander(f"Step {i}", expanded=(i == len(snippets))):
                    st.code(snippet, language="markdown")
            
            st.caption(f"*{len(snippets)} step(s) generated. Click 'Show Next Step' for more.*")
        else:
            st.info("👆 Click 'Show Next Step' to generate the first visual guide.")
    else:
        st.info("💡 Start a conversation about your taxes, then click 'Show Next Step' to see form mapping visuals.")

# ==========================================
# IRS Document Search Section
# ==========================================
def render_search_section():
    st.divider()
    with st.expander("📚 Search IRS Documents", expanded=False):
        st.markdown("Search the IRS document database for specific form information.")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "Search query:",
                placeholder="e.g., W-2 Box 2 federal withholding",
                key="irs_search_input"
            )
        
        with col2:
            search_btn = st.button("🔍 Search", key="irs_search_btn", use_container_width=True)
        
        if search_btn and search_query:
            with st.spinner("Searching IRS documents..."):
                result = st.session_state.orchestrator.rag_agent.search(search_query, k=3)
                st.session_state.search_result = result
        
        if st.session_state.get('search_result'):
            st.markdown("### 📄 Search Results")
            st.markdown(st.session_state.search_result)
            
            if st.button("🗑️ Clear Results", key="clear_search"):
                st.session_state.search_result = None
                st.rerun()

# ==========================================
# Main Application
# ==========================================
def main():
    st.title("🏛️ AI Tax Assistant")
    st.caption("Powered by Google Gemini 2.5 Pro + LangChain + RAG with Visual Form Mapping")
    
    # ==========================================
    # Legal Disclaimer Banner (Always Visible)
    # ==========================================
    st.markdown("""
    <div class="disclaimer-banner">
    ⚠️ <strong>IMPORTANT:</strong> This AI assistant is for <strong>educational purposes only</strong>. 
    It is NOT a substitute for professional tax advice from a CPA or tax attorney. 
    You are solely responsible for the accuracy of your tax filings. 
    <a href="#disclaimer-details">Read full disclaimer</a>
    </div>
    """, unsafe_allow_html=True)
    
    # Full disclaimer in expander
    with st.expander("📜 Full Legal Disclaimer & Privacy Notice", expanded=False):
        tab1, tab2 = st.tabs(["⚖️ Legal Disclaimer", "🔒 Privacy Notice"])
        
        with tab1:
            st.markdown(LEGAL_DISCLAIMER)
        
        with tab2:
            st.markdown(PRIVACY_NOTICE)
        
        # Acknowledgment checkbox (optional but recommended)
        if 'disclaimer_acknowledged' not in st.session_state:
            st.session_state.disclaimer_acknowledged = False
        
        acknowledged = st.checkbox(
            "I understand this tool is for educational purposes only and does not constitute professional tax advice.",
            value=st.session_state.disclaimer_acknowledged,
            key="disclaimer_checkbox"
        )
        st.session_state.disclaimer_acknowledged = acknowledged
    
    # Initialize API key
    if 'api_key' not in st.session_state:
        st.session_state.api_key = os.environ.get("GOOGLE_API_KEY", "")
        try:
            if st.secrets.get("GOOGLE_API_KEY"):
                st.session_state.api_key = st.secrets["GOOGLE_API_KEY"]
        except:
            pass
    
    # Check for API key
    if not st.session_state.api_key:
        st.error("⚠️ Please set GOOGLE_API_KEY in environment variables or Streamlit secrets.")
        st.session_state.api_key = st.text_input("Enter Google API Key:", type="password")
        if not st.session_state.api_key:
            st.stop()
    
    # ==========================================
    # Initialize Session Memory
    # ==========================================
    memory_manager = load_memory_manager()
    
    # Get or create session ID
    if 'session_id' not in st.session_state:
        st.session_state.session_id = get_or_create_session_id()
    
    # Load session from memory
    if 'session_loaded' not in st.session_state:
        session = load_session_from_memory(memory_manager, st.session_state.session_id)
        st.session_state.user_session = session
        
        # Load existing data from memory
        load_session_state_from_memory(memory_manager, session)
        st.session_state.session_loaded = True
    
    # Initialize system
    if 'orchestrator' not in st.session_state:
        with st.spinner("🔧 Initializing AI Tax Assistant..."):
            try:
                st.session_state.orchestrator = TaxOrchestrator(st.session_state.api_key)
                st.session_state.user_profile = UserProfile()
                st.session_state.messages = []
                st.session_state.checklist = []
                st.session_state.current_visual_topic = None
                st.session_state.latest_visual_snippet = None
                st.session_state.uploaded_doc_text = None
                st.session_state.uploaded_img_text = None
                st.session_state.uploaded_doc_name = None
                st.session_state.uploaded_img_name = None
                st.session_state.search_result = None
                st.success("✅ System ready!")
            except Exception as e:
                st.error(f"❌ Initialization failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.stop()
    
    # Render sidebar
    render_sidebar()
    
    # Active documents indicator
    if st.session_state.get('uploaded_doc_text') or st.session_state.get('uploaded_img_text'):
        active_docs = []
        if st.session_state.get('uploaded_doc_name'):
            active_docs.append(f"📄 {st.session_state.uploaded_doc_name}")
        if st.session_state.get('uploaded_img_name'):
            active_docs.append(f"🖼️ {st.session_state.uploaded_img_name}")
        st.info(f"📎 Active documents: {', '.join(active_docs)}")
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your taxes..."):
        # Check for PII in user input
        detected_pii = PIIHandler.detect_pii(prompt)
        masked_prompt, pii_counts = PIIHandler.mask_pii(prompt)
        
        # Build context with uploaded documents
        context_parts = []
        
        if st.session_state.get('uploaded_doc_text'):
            context_parts.append(f"[Document: {st.session_state.uploaded_doc_name}]\n{st.session_state.uploaded_doc_text[:2000]}")
        
        if st.session_state.get('uploaded_img_text'):
            context_parts.append(f"[OCR from: {st.session_state.uploaded_img_name}]\n{st.session_state.uploaded_img_text}")
        
        full_prompt = "\n\n".join(context_parts) + f"\n\nUser Question: {masked_prompt}" if context_parts else masked_prompt
        display_prompt = prompt
        
        # Display user message
        st.session_state.messages.append({"role": "user", "content": display_prompt})
        with st.chat_message("user"):
            st.markdown(display_prompt)
            
            # Show PII warning if detected in user input
            if pii_counts:
                st.markdown(f"""
                <div class="pii-warning">
                🔒 <strong>Privacy Protection:</strong> We detected and masked {sum(pii_counts.values())} 
                sensitive item(s) in your message (SSN, account numbers, etc.) before processing.
                </div>
                """, unsafe_allow_html=True)
        
        # Extract user info (Intake Agent)
        with st.status("🔍 Analyzing...", expanded=False) as status:
            try:
                new_data = st.session_state.orchestrator.run_intake(full_prompt)
                current_data = st.session_state.user_profile.dict()
                current_data.update(new_data.dict(exclude_none=True))
                st.session_state.user_profile = UserProfile(**current_data)
                status.update(label="✅ Info extracted!", state="complete")
            except Exception as e:
                status.update(label="⚠️ Partial extraction", state="running")
        
        # Generate response
        with st.chat_message("assistant"):
            with st.status("🤖 Thinking...", expanded=True) as status:
                try:
                    response = st.session_state.orchestrator.run_orchestrator(
                        full_prompt,
                        st.session_state.user_profile
                    )
                    answer = response["output"]
                    status.update(label="✅ Done!", state="complete")
                    st.markdown(answer)
                    
                    # Add reminder disclaimer after response
                    st.caption("*⚠️ Remember: This is educational guidance only, not professional tax advice.*")
                    
                    # Clear used documents
                    if context_parts:
                        st.session_state.uploaded_doc_text = None
                        st.session_state.uploaded_img_text = None
                        st.session_state.uploaded_doc_name = None
                        st.session_state.uploaded_img_name = None
                except Exception as e:
                    answer = f"❌ Error: {str(e)}"
                    st.error(answer)
                    status.update(label="❌ Error", state="error")
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # ==========================================
        # Sync to Persistent Memory
        # ==========================================
        sync_session_to_memory(
            memory_manager,
            st.session_state.user_session,
            st.session_state.user_profile,
            st.session_state.checklist,
            st.session_state.messages
        )
        
        # Update checklist
        with st.status("📋 Updating checklist...", expanded=False) as checklist_status:
            try:
                checklist = st.session_state.orchestrator.generate_checklist(
                    st.session_state.messages,
                    st.session_state.user_profile
                )
                st.session_state.checklist = checklist
                checklist_status.update(label="✅ Checklist updated!", state="complete")
            except Exception as e:
                checklist_status.update(label="⚠️ Checklist update failed", state="error")
        
        # Final sync to memory after checklist update
        sync_session_to_memory(
            memory_manager,
            st.session_state.user_session,
            st.session_state.user_profile,
            st.session_state.checklist,
            st.session_state.messages
        )
        
        st.rerun()
    
    # Render Visual Help Section (RAG-Enhanced)
    render_visual_help()
    
    # Render IRS Document Search
    render_search_section()

if __name__ == "__main__":
    main()