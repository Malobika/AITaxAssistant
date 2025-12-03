"""
Session Memory Module for AI Tax Assistant
Provides persistent storage for user sessions, profiles, and conversation history
using ChromaDB vector database.
"""

import os
import sys
import json
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any
import uuid

# Fix sqlite3 issue
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except (ImportError, KeyError):
    pass

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pydantic import BaseModel, Field

# ==========================================
# Configuration
# ==========================================
MEMORY_DB_DIRECTORY = "user_session_memory_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Collection names
USER_PROFILES_COLLECTION = "user_profiles"
CONVERSATION_HISTORY_COLLECTION = "conversation_history"
CHECKLIST_PROGRESS_COLLECTION = "checklist_progress"
FORM_DATA_COLLECTION = "form_field_data"

# ==========================================
# Data Models
# ==========================================
class UserSession(BaseModel):
    """Represents a user session"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    last_active: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    # Profile data
    citizenship_status: Optional[str] = None
    student_status: Optional[str] = None
    employment_details: Optional[str] = None
    tax_filing_experience: Optional[str] = None
    residency_duration: Optional[str] = None
    income: Optional[int] = None
    residency_state: Optional[str] = None
    filing_status: Optional[str] = None
    
    # Session metadata
    profile_completion: float = 0.0
    checklist_completion: float = 0.0
    forms_completed: List[str] = Field(default_factory=list)
    current_form: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserSession":
        return cls(**data)


class ConversationMessage(BaseModel):
    """Represents a single conversation message"""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FormFieldData(BaseModel):
    """Represents form field data entered by user"""
    field_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    form_name: str  # e.g., "W-2", "1040-NR", "1098-T"
    field_name: str  # e.g., "Box 1", "Line 1a"
    field_value: str  # Masked value
    original_value_hash: str  # Hash of original for verification
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    is_verified: bool = False


# ==========================================
# Session Memory Manager
# ==========================================
class SessionMemoryManager:
    """
    Manages persistent session memory using ChromaDB
    Stores: user profiles, conversation history, checklist progress, form data
    """
    
    def __init__(self, db_directory: str = MEMORY_DB_DIRECTORY):
        self.db_directory = db_directory
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        # Initialize collections
        self._init_collections()
        print("✅ Session Memory Manager initialized")
    
    def _init_collections(self):
        """Initialize all ChromaDB collections"""
        
        # User Profiles Collection
        self.profiles_db = Chroma(
            persist_directory=os.path.join(self.db_directory, "profiles"),
            embedding_function=self.embeddings,
            collection_name=USER_PROFILES_COLLECTION,
            collection_metadata={"description": "User profiles and session data"}
        )
        
        # Conversation History Collection
        self.conversations_db = Chroma(
            persist_directory=os.path.join(self.db_directory, "conversations"),
            embedding_function=self.embeddings,
            collection_name=CONVERSATION_HISTORY_COLLECTION,
            collection_metadata={"description": "Conversation history"}
        )
        
        # Checklist Progress Collection
        self.checklist_db = Chroma(
            persist_directory=os.path.join(self.db_directory, "checklists"),
            embedding_function=self.embeddings,
            collection_name=CHECKLIST_PROGRESS_COLLECTION,
            collection_metadata={"description": "Checklist progress tracking"}
        )
        
        # Form Field Data Collection
        self.form_data_db = Chroma(
            persist_directory=os.path.join(self.db_directory, "form_data"),
            embedding_function=self.embeddings,
            collection_name=FORM_DATA_COLLECTION,
            collection_metadata={"description": "Form field data (masked)"}
        )
    
    # ==========================================
    # Session Management
    # ==========================================
    
    def create_session(self, user_id: Optional[str] = None) -> UserSession:
        """Create a new user session"""
        session = UserSession(user_id=user_id or str(uuid.uuid4()))
        self.save_session(session)
        return session
    
    def save_session(self, session: UserSession):
        """Save or update a user session"""
        session.last_active = datetime.now().isoformat()
        
        # Create searchable text for the session
        session_text = f"""
        User Session: {session.session_id}
        User ID: {session.user_id}
        Citizenship: {session.citizenship_status or 'Unknown'}
        Student Status: {session.student_status or 'Unknown'}
        Employment: {session.employment_details or 'Unknown'}
        Income: {session.income or 'Unknown'}
        State: {session.residency_state or 'Unknown'}
        Filing Status: {session.filing_status or 'Unknown'}
        Profile Completion: {session.profile_completion}%
        Checklist Completion: {session.checklist_completion}%
        """
        
        metadata = session.to_dict()
        # Convert lists to JSON strings for ChromaDB compatibility
        metadata['forms_completed'] = json.dumps(metadata.get('forms_completed', []))
        
        # Check if session exists
        existing = self._get_session_by_id(session.session_id)
        
        if existing:
            # Update existing - delete and re-add
            self.profiles_db._collection.delete(
                where={"session_id": session.session_id}
            )
        
        self.profiles_db.add_texts(
            texts=[session_text],
            metadatas=[metadata],
            ids=[session.session_id]
        )
    
    def _get_session_by_id(self, session_id: str) -> Optional[UserSession]:
        """Get session by ID"""
        try:
            results = self.profiles_db._collection.get(
                ids=[session_id],
                include=["metadatas"]
            )
            
            if results and results['metadatas']:
                metadata = results['metadatas'][0]
                # Convert JSON strings back to lists
                if 'forms_completed' in metadata:
                    metadata['forms_completed'] = json.loads(metadata['forms_completed'])
                return UserSession.from_dict(metadata)
        except Exception as e:
            print(f"Error getting session: {e}")
        
        return None
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get a session by ID"""
        return self._get_session_by_id(session_id)
    
    def get_or_create_session(self, session_id: Optional[str] = None, 
                               user_id: Optional[str] = None) -> UserSession:
        """Get existing session or create new one"""
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        
        return self.create_session(user_id)
    
    def find_sessions_by_user(self, user_id: str) -> List[UserSession]:
        """Find all sessions for a user"""
        try:
            results = self.profiles_db._collection.get(
                where={"user_id": user_id},
                include=["metadatas"]
            )
            
            sessions = []
            if results and results['metadatas']:
                for metadata in results['metadatas']:
                    if 'forms_completed' in metadata:
                        metadata['forms_completed'] = json.loads(metadata['forms_completed'])
                    sessions.append(UserSession.from_dict(metadata))
            
            return sessions
        except Exception as e:
            print(f"Error finding sessions: {e}")
            return []
    
    def search_similar_sessions(self, query: str, k: int = 5) -> List[UserSession]:
        """Search for similar sessions based on query"""
        try:
            results = self.profiles_db.similarity_search(query, k=k)
            
            sessions = []
            for doc in results:
                metadata = doc.metadata
                if 'forms_completed' in metadata:
                    metadata['forms_completed'] = json.loads(metadata['forms_completed'])
                sessions.append(UserSession.from_dict(metadata))
            
            return sessions
        except Exception as e:
            print(f"Error searching sessions: {e}")
            return []
    
    # ==========================================
    # Conversation History
    # ==========================================
    
    def save_message(self, session_id: str, role: str, content: str, 
                     metadata: Dict[str, Any] = None):
        """Save a conversation message"""
        message = ConversationMessage(
            session_id=session_id,
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        # Create searchable text
        message_text = f"{role}: {content}"
        
        msg_metadata = {
            "message_id": message.message_id,
            "session_id": session_id,
            "role": role,
            "timestamp": message.timestamp,
            "metadata": json.dumps(message.metadata)
        }
        
        self.conversations_db.add_texts(
            texts=[message_text],
            metadatas=[msg_metadata],
            ids=[message.message_id]
        )
    
    def get_conversation_history(self, session_id: str, 
                                  limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        try:
            results = self.conversations_db._collection.get(
                where={"session_id": session_id},
                include=["documents", "metadatas"]
            )
            
            if not results or not results['metadatas']:
                return []
            
            messages = []
            for i, metadata in enumerate(results['metadatas']):
                messages.append({
                    "role": metadata.get("role", "user"),
                    "content": results['documents'][i] if results['documents'] else "",
                    "timestamp": metadata.get("timestamp", ""),
                    "metadata": json.loads(metadata.get("metadata", "{}"))
                })
            
            # Sort by timestamp
            messages.sort(key=lambda x: x.get("timestamp", ""))
            
            return messages[-limit:]
        except Exception as e:
            print(f"Error getting conversation history: {e}")
            return []
    
    def search_conversations(self, session_id: str, query: str, 
                             k: int = 5) -> List[Dict[str, Any]]:
        """Search conversation history for relevant messages"""
        try:
            results = self.conversations_db.similarity_search(
                query, 
                k=k,
                filter={"session_id": session_id}
            )
            
            messages = []
            for doc in results:
                messages.append({
                    "role": doc.metadata.get("role", "user"),
                    "content": doc.page_content,
                    "timestamp": doc.metadata.get("timestamp", ""),
                })
            
            return messages
        except Exception as e:
            print(f"Error searching conversations: {e}")
            return []
    
    # ==========================================
    # Checklist Progress
    # ==========================================
    
    def save_checklist(self, session_id: str, checklist: List[Dict[str, Any]]):
        """Save checklist progress"""
        checklist_text = json.dumps(checklist, indent=2)
        
        # Calculate overall completion
        total_completion = 0
        if checklist:
            total_completion = sum(s.get('completion', 0) for s in checklist) / len(checklist)
        
        metadata = {
            "session_id": session_id,
            "checklist_json": checklist_text,
            "total_completion": total_completion,
            "num_sections": len(checklist),
            "timestamp": datetime.now().isoformat()
        }
        
        # Delete existing checklist for this session
        try:
            self.checklist_db._collection.delete(
                where={"session_id": session_id}
            )
        except:
            pass
        
        self.checklist_db.add_texts(
            texts=[f"Checklist for session {session_id}: {total_completion}% complete"],
            metadatas=[metadata],
            ids=[f"checklist_{session_id}"]
        )
    
    def get_checklist(self, session_id: str) -> List[Dict[str, Any]]:
        """Get checklist for a session"""
        try:
            results = self.checklist_db._collection.get(
                ids=[f"checklist_{session_id}"],
                include=["metadatas"]
            )
            
            if results and results['metadatas']:
                checklist_json = results['metadatas'][0].get('checklist_json', '[]')
                return json.loads(checklist_json)
        except Exception as e:
            print(f"Error getting checklist: {e}")
        
        return []
    
    # ==========================================
    # Form Field Data
    # ==========================================
    
    def save_form_field(self, session_id: str, form_name: str, 
                        field_name: str, field_value: str):
        """Save a form field value (masked)"""
        # Hash the original value for verification later
        value_hash = hashlib.sha256(field_value.encode()).hexdigest()[:16]
        
        field_data = FormFieldData(
            session_id=session_id,
            form_name=form_name,
            field_name=field_name,
            field_value=field_value,  # Already masked by PIIHandler
            original_value_hash=value_hash
        )
        
        field_text = f"{form_name} {field_name}: {field_value}"
        
        metadata = {
            "field_id": field_data.field_id,
            "session_id": session_id,
            "form_name": form_name,
            "field_name": field_name,
            "field_value": field_value,
            "value_hash": value_hash,
            "timestamp": field_data.timestamp,
            "is_verified": False
        }
        
        self.form_data_db.add_texts(
            texts=[field_text],
            metadatas=[metadata],
            ids=[field_data.field_id]
        )
    
    def get_form_fields(self, session_id: str, 
                        form_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all form fields for a session"""
        try:
            where_filter = {"session_id": session_id}
            if form_name:
                where_filter["form_name"] = form_name
            
            results = self.form_data_db._collection.get(
                where=where_filter,
                include=["metadatas"]
            )
            
            if results and results['metadatas']:
                return results['metadatas']
        except Exception as e:
            print(f"Error getting form fields: {e}")
        
        return []
    
    def search_form_fields(self, session_id: str, query: str, 
                           k: int = 5) -> List[Dict[str, Any]]:
        """Search form fields by query"""
        try:
            results = self.form_data_db.similarity_search(
                query,
                k=k,
                filter={"session_id": session_id}
            )
            
            return [doc.metadata for doc in results]
        except Exception as e:
            print(f"Error searching form fields: {e}")
            return []
    
    # ==========================================
    # Session Analytics
    # ==========================================
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of a session's progress"""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        conversation_count = len(self.get_conversation_history(session_id, limit=1000))
        checklist = self.get_checklist(session_id)
        form_fields = self.get_form_fields(session_id)
        
        checklist_completion = 0
        if checklist:
            checklist_completion = sum(s.get('completion', 0) for s in checklist) / len(checklist)
        
        return {
            "session_id": session_id,
            "user_id": session.user_id,
            "created_at": session.created_at,
            "last_active": session.last_active,
            "profile_completion": session.profile_completion,
            "checklist_completion": checklist_completion,
            "conversation_count": conversation_count,
            "forms_started": list(set(f.get('form_name', '') for f in form_fields)),
            "fields_filled": len(form_fields),
            "profile": {
                "citizenship": session.citizenship_status,
                "student_status": session.student_status,
                "employment": session.employment_details,
                "income": session.income,
                "state": session.residency_state,
            }
        }
    
    # ==========================================
    # Cleanup
    # ==========================================
    
    def delete_session(self, session_id: str):
        """Delete all data for a session"""
        try:
            # Delete from all collections
            self.profiles_db._collection.delete(ids=[session_id])
            self.conversations_db._collection.delete(where={"session_id": session_id})
            self.checklist_db._collection.delete(ids=[f"checklist_{session_id}"])
            self.form_data_db._collection.delete(where={"session_id": session_id})
            print(f"✅ Session {session_id} deleted")
        except Exception as e:
            print(f"Error deleting session: {e}")
    
    def clear_old_sessions(self, days_old: int = 30):
        """Clear sessions older than specified days"""
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=days_old)).isoformat()
        
        try:
            # Get all sessions
            results = self.profiles_db._collection.get(include=["metadatas"])
            
            if results and results['metadatas']:
                for i, metadata in enumerate(results['metadatas']):
                    last_active = metadata.get('last_active', '')
                    if last_active < cutoff:
                        session_id = metadata.get('session_id')
                        if session_id:
                            self.delete_session(session_id)
                            print(f"🗑️ Cleared old session: {session_id}")
        except Exception as e:
            print(f"Error clearing old sessions: {e}")


# ==========================================
# Convenience Functions
# ==========================================

def get_memory_manager() -> SessionMemoryManager:
    """Get or create a singleton memory manager"""
    if not hasattr(get_memory_manager, '_instance'):
        get_memory_manager._instance = SessionMemoryManager()
    return get_memory_manager._instance


def generate_session_id_from_browser(browser_fingerprint: str) -> str:
    """Generate a consistent session ID from browser fingerprint"""
    return hashlib.sha256(browser_fingerprint.encode()).hexdigest()[:32]
