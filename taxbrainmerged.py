import os
import sys

# Fix sqlite3 issue
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except (ImportError, KeyError):
    pass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import Tool
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import json
import re

# ==========================================
# Configuration
# ==========================================
DB_DIRECTORY = "federal_tax_vector_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "federal_tax_documents"

# ==========================================
# Data Structures
# ==========================================
class UserProfile(BaseModel):
    """User's complete tax profile"""
    citizenship_status: Optional[str] = Field(default=None)
    student_status: Optional[str] = Field(default=None)
    employment_details: Optional[str] = Field(default=None)
    tax_filing_experience: Optional[str] = Field(default=None)
    residency_duration: Optional[str] = Field(default=None)
    income: Optional[int] = Field(default=None)
    residency_state: Optional[str] = Field(default=None)
    name: Optional[str] = Field(default=None)
    filing_status: Optional[str] = Field(default=None)
    w2_forms_count: Optional[int] = Field(default=None)

# ==========================================
# 1. Intake Agent
# ==========================================
class IntakeAgent:
    """Hybrid Intake Agent with auto-extraction and friendly dialogue"""
    
    QUESTIONNAIRE = [
        "What is your citizenship status? (US Citizen / Green Card Holder / International Student / Other)",
        "Are you a student? (Full-time / Part-time / Not a student)",
        "What is your employment status? (On-campus job / Off-campus job / Self-employed / Multiple jobs)",
        "Have you filed US taxes before? (Yes / No)",
        "How long have you lived in your current state?",
        "What was your total income last year? (Approximate)",
        "Which state do you currently live in?"
    ]
    
    CONVERSATIONAL_PROMPT = """I'm your AI tax assistant! 👋

I notice you need help with your taxes. To give you the best advice, I'd like to learn a bit about your situation.

You can either:
1. **Answer these quick questions:**
{questions}

2. **Or just tell me naturally**, like:
   - "I'm an international student on F-1 visa, working on-campus, earned $15k"
   - "I'm a working professional in California, made $60k last year"

I'll understand either way! 😊"""
    
    def __init__(self, llm):
        self.llm = llm
        self.extractor = llm.with_structured_output(UserProfile)
    
    def get_questionnaire(self) -> str:
        questions = "\n".join([f"   • {q}" for q in self.QUESTIONNAIRE])
        return self.CONVERSATIONAL_PROMPT.format(questions=questions)
    
    def extract_info(self, user_input: str) -> UserProfile:
        try:
            return self.extractor.invoke(user_input)
        except Exception as e:
            print(f"⚠️ Intake extraction failed: {e}")
            return UserProfile()
    
    def check_completeness(self, profile: UserProfile) -> Dict[str, Any]:
        required = ['citizenship_status', 'student_status', 'employment_details',
                   'tax_filing_experience', 'income', 'residency_state']
        missing = [f for f in required if getattr(profile, f) is None]
        return {
            'complete': len(missing) == 0,
            'missing_fields': missing,
            'completion_rate': (len(required) - len(missing)) / len(required) * 100
        }
    
    def get_smart_followup(self, profile: UserProfile) -> str:
        completeness = self.check_completeness(profile)
        if completeness['complete']:
            return "✅ Perfect! I have everything I need. What would you like help with today?"
        
        if completeness['completion_rate'] == 0:
            return self.get_questionnaire()
        
        friendly_questions = {
            'citizenship_status': "your citizenship status",
            'student_status': "if you're currently a student",
            'employment_details': "your employment situation",
            'income': "your approximate income last year",
            'residency_state': "which state you live in",
            'tax_filing_experience': "if you've filed US taxes before",
        }
        
        missing = completeness['missing_fields'][:3]
        missing_text = ", ".join([friendly_questions.get(f, f) for f in missing])
        
        return f"""Great! I've got some of your info ({completeness['completion_rate']:.0f}% complete).

To give you better guidance, I'd like to know {missing_text}.

**Or just ask your tax question directly!** 🚀"""

# ==========================================
# 2. RAG Agent - With Enhanced Visual Support
# ==========================================
class RAGAgent:
    """RAG Agent with ChromaDB retrieval and visual mapping support"""
    
    def __init__(self, llm):
        self.llm = llm
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        if os.path.exists(DB_DIRECTORY):
            self.db = Chroma(
                persist_directory=DB_DIRECTORY,
                embedding_function=self.embeddings,
                collection_name=COLLECTION_NAME
            )
            print("✅ RAG Agent: Connected to ChromaDB")
        else:
            print("⚠️ RAG Agent: ChromaDB not found")
            self.db = None
        
        self._build_qa_chain()
    
    def _build_qa_chain(self):
        if not self.db:
            self.qa_chain = None
            return
        
        template = """You are a tax expert assistant. Answer based on IRS documentation.

User Profile:
- Citizenship: {citizenship_status}
- Student Status: {student_status}
- Employment: {employment_details}
- Income: ${income}
- State: {residency_state}

IRS Documentation:
{context}

User Question: {question}

Provide a clear, helpful answer tailored to this user's situation."""

        prompt = ChatPromptTemplate.from_template(template)
        
        def retrieve_and_format(inputs):
            query = inputs["question"]
            docs = self.db.similarity_search(query, k=3)
            if not docs:
                return "No relevant information found."
            
            formatted = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source_file', 'Unknown')
                form = doc.metadata.get('form_number', 'N/A')
                content = doc.page_content[:400]
                formatted.append(f"Source {i} - {source} (Form {form}):\n{content}...")
            return "\n\n".join(formatted)
        
        self.qa_chain = (
            {
                "context": retrieve_and_format,
                "question": lambda x: x["question"],
                "citizenship_status": lambda x: x.get("citizenship_status", "Unknown"),
                "student_status": lambda x: x.get("student_status", "Unknown"),
                "employment_details": lambda x: x.get("employment_details", "Unknown"),
                "income": lambda x: x.get("income", "Unknown"),
                "residency_state": lambda x: x.get("residency_state", "Unknown"),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def search(self, query: str, doc_type: str = "all", k: int = 3) -> str:
        """Basic search returning formatted results"""
        if not self.db:
            return "Tax database is not available."
        
        try:
            filter_dict = {"doc_type": doc_type} if doc_type != "all" else None
            results = self.db.similarity_search(query, k=k, filter=filter_dict)
            
            if not results:
                return "No relevant information found."
            
            response = "Information from IRS Documents:\n\n"
            for i, doc in enumerate(results, 1):
                source = doc.metadata.get('source_file', 'Unknown')
                form = doc.metadata.get('form_number', 'N/A')
                content = doc.page_content[:300]
                response += f"Source {i} - {source} (Form {form}):\n{content}...\n\n"
            return response
        except Exception as e:
            return f"Error searching database: {str(e)}"
    
    def search_form_mapping(self, source_form: str, target_form: str, field: str = None) -> str:
        """
        Search for specific form-to-form field mappings
        E.g., W-2 Box 1 -> Form 1040-NR Line 1a
        """
        if not self.db:
            return "Tax database is not available."
        
        # Build targeted query
        if field:
            query = f"{source_form} {field} to {target_form} mapping instructions"
        else:
            query = f"{source_form} to {target_form} field mapping instructions"
        
        try:
            # Search with form-specific filters
            results = self.db.similarity_search(query, k=5)
            
            if not results:
                return f"No mapping information found for {source_form} to {target_form}."
            
            # Format for visual generation
            mapping_info = []
            for doc in results:
                source = doc.metadata.get('source_file', 'Unknown')
                form_num = doc.metadata.get('form_number', 'N/A')
                content = doc.page_content[:500]
                mapping_info.append({
                    "source": source,
                    "form": form_num,
                    "content": content
                })
            
            return mapping_info
        except Exception as e:
            return f"Error searching mappings: {str(e)}"
    
    def answer_with_context(self, query: str, user_profile: UserProfile) -> str:
        if not self.qa_chain:
            prompt = f"""You are a tax expert assistant.
User Profile: {user_profile.dict(exclude_none=True)}
Question: {query}
Provide helpful tax guidance."""
            response = self.llm.invoke(prompt)
            return response.content
        
        try:
            chain_input = {
                "question": query,
                "citizenship_status": user_profile.citizenship_status or "Unknown",
                "student_status": user_profile.student_status or "Unknown",
                "employment_details": user_profile.employment_details or "Unknown",
                "income": user_profile.income or "Unknown",
                "residency_state": user_profile.residency_state or "Unknown",
            }
            return self.qa_chain.invoke(chain_input)
        except Exception as e:
            return f"Error generating answer: {str(e)}"

# ==========================================
# 3. Tool Agent
# ==========================================
class ToolAgent:
    """Tax calculation tools"""
    
    @staticmethod
    def calculate_tax(income: int, filing_status: str = "single") -> str:
        standard_deductions = {
            "single": 14600, "married_jointly": 29200,
            "married_separately": 14600, "head_of_household": 21900
        }
        
        tax_brackets = [
            (11600, 0.10), (47150, 0.12), (100525, 0.22),
            (191950, 0.24), (243725, 0.32), (609350, 0.35), (float('inf'), 0.37)
        ]
        
        status = filing_status.lower().replace(" ", "_")
        deduction = standard_deductions.get(status, 14600)
        taxable_income = max(0, income - deduction)
        
        tax = 0
        prev_bracket = 0
        for bracket, rate in tax_brackets:
            if taxable_income <= bracket:
                tax += (taxable_income - prev_bracket) * rate
                break
            else:
                tax += (bracket - prev_bracket) * rate
                prev_bracket = bracket
        
        effective_rate = round((tax / income * 100), 2) if income > 0 else 0
        
        return f"""Tax Calculation Results:
- Gross Income: ${income:,}
- Standard Deduction: ${deduction:,}
- Taxable Income: ${taxable_income:,}
- Estimated Tax: ${round(tax, 2):,}
- Effective Tax Rate: {effective_rate}%

This is an estimate based on 2024 federal tax rates."""

# ==========================================
# 4. Visual Agent - NEW: RAG-Enhanced Visuals
# ==========================================
class VisualAgent:
    """
    Generates step-by-step visual guides for form mappings
    Uses RAG to retrieve accurate IRS documentation
    """
    
    def __init__(self, llm, rag_agent: RAGAgent):
        self.llm = llm
        self.rag = rag_agent
        self.generated_snippets = {}  # {topic: [snippet1, snippet2, ...]}
    
    def infer_topic(self, messages: List[dict], user_profile: UserProfile) -> str:
        """Infer the most relevant visual topic from conversation"""
        recent_text = "\n".join([
            f"{m.get('role', 'user')}: {m.get('content', '')}" 
            for m in messages[-10:]
        ])
        
        profile_str = json.dumps(user_profile.dict(exclude_none=True), indent=2)
        
        prompt = f"""Based on this tax conversation, determine the most relevant form mapping topic.

Conversation:
{recent_text or "[no messages yet]"}

User Profile:
{profile_str}

Return EXACTLY ONE topic key from these options:
- w2_to_1040nr (W-2 to Form 1040-NR for nonresidents)
- w2_to_1040 (W-2 to Form 1040 for residents)
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
- Default: w2_to_1040nr for students, w2_to_1040 for professionals

Respond with ONLY the topic key, nothing else."""

        try:
            response = self.llm.invoke(prompt)
            topic = response.content.strip().lower().replace("-", "_")
            # Validate topic
            valid_topics = [
                "w2_to_1040nr", "w2_to_1040", "1098t_to_1040nr", "1098t_to_1040",
                "1099int_to_1040", "1099nec_to_schedule_c", "schedule1_adjustments",
                "generic_tax_visual"
            ]
            if topic not in valid_topics:
                topic = "w2_to_1040nr"
            return topic
        except Exception as e:
            print(f"⚠️ Topic inference failed: {e}")
            return "w2_to_1040nr"
    
    def _parse_topic(self, topic: str) -> Dict[str, str]:
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
    
    def generate_visual_snippet(self, topic: str, user_profile: UserProfile) -> str:
        """
        Generate the NEXT visual snippet for a topic using RAG
        """
        existing = self.generated_snippets.get(topic, [])
        step_number = len(existing) + 1
        
        # Parse topic to get source/target forms
        forms = self._parse_topic(topic)
        source_form = forms["source"]
        target_form = forms["target"]
        
        # ========== RAG INTEGRATION ==========
        # Query ChromaDB for relevant form mapping information
        rag_context = ""
        if self.rag and self.rag.db:
            # Build step-specific query
            step_queries = {
                1: f"{source_form} Box 1 wages {target_form}",
                2: f"{source_form} Box 2 federal tax withheld {target_form}",
                3: f"{source_form} Box 3 4 Social Security {target_form}",
                4: f"{source_form} Box 5 6 Medicare {target_form}",
                5: f"{source_form} Box 12 14 other information {target_form}",
            }
            query = step_queries.get(step_number, f"{source_form} to {target_form} mapping step {step_number}")
            
            try:
                docs = self.rag.db.similarity_search(query, k=2)
                if docs:
                    rag_context = "\n\n".join([
                        f"IRS Reference ({doc.metadata.get('source_file', 'Unknown')}):\n{doc.page_content[:300]}"
                        for doc in docs
                    ])
            except Exception as e:
                print(f"⚠️ RAG search failed: {e}")
        # =====================================
        
        profile_str = json.dumps(user_profile.dict(exclude_none=True), indent=2)
        
        system_prompt = """You are a tax visualization expert. Create step-by-step visual guides 
showing how to map values from source tax forms to destination forms.

Your output should be a code-style text block with:
- Clear header with step number and focus
- Box-to-line mappings using arrows (→)
- Specific box numbers and line numbers
- Brief explanations
- Example values where helpful"""

        user_prompt = f"""Create Step {step_number} of a visual guide for: {source_form} → {target_form}

User Profile:
{profile_str}

{"IRS Documentation Reference:" + chr(10) + rag_context if rag_context else ""}

Previous steps completed: {step_number - 1}

Requirements:
1. Start with a header block like:
   📋 {source_form} → {target_form} Mapping (Step {step_number}/5)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Focus: [specific focus for this step]

2. Show 1-2 specific box-to-line mappings with arrows
3. Include brief explanation of what each value represents
4. Add example if helpful
5. End with a separator line
6. Keep under 150 words

For step {step_number}, focus on:
- Step 1: Wages/compensation (Box 1)
- Step 2: Federal tax withheld (Box 2)  
- Step 3: Social Security (Boxes 3-4)
- Step 4: Medicare (Boxes 5-6)
- Step 5: Other codes and state info (Boxes 12, 14)"""

        try:
            response = self.llm.invoke(f"{system_prompt}\n\n{user_prompt}")
            snippet = response.content.strip()
            
            # Store the snippet
            if topic not in self.generated_snippets:
                self.generated_snippets[topic] = []
            self.generated_snippets[topic].append(snippet)
            
            return snippet
        except Exception as e:
            return f"Error generating visual: {str(e)}"
    
    def get_all_snippets(self, topic: str) -> List[str]:
        """Get all generated snippets for a topic"""
        return self.generated_snippets.get(topic, [])
    
    def reset_topic(self, topic: str = None):
        """Reset snippets for a topic or all topics"""
        if topic:
            self.generated_snippets[topic] = []
        else:
            self.generated_snippets = {}

# ==========================================
# 5. Checklist Agent
# ==========================================
class ChecklistAgent:
    """Generates and maintains tax filing checklist"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def generate_checklist(self, conversation_history: List[dict], user_profile: UserProfile) -> List[dict]:
        if not conversation_history:
            return []
        
        convo_text = "\n".join([
            f"{msg.get('role', 'user').upper()}: {msg.get('content', '')}" 
            for msg in conversation_history
        ])
        
        system_prompt = """You are a CHECKLIST AGENT for US tax filing.

Return ONLY valid JSON in this format:
{
  "sections": [
    {
      "heading": "Collect W-2 forms",
      "status": "pending",
      "details": [
        {"item": "Collect W-2 from each employer", "status": "done"},
        {"item": "Record wages (Box 1)", "status": "pending"}
      ]
    }
  ]
}

Rules:
- ACTION headings (e.g., "Collect W-2 forms", "Complete Form 1040-NR")
- 3-7 detailed sub-items per section
- Mark "done" ONLY if user explicitly mentioned completing it
- Tailor to profile (student → 1098-T, professional → W-2/1099)
- 4-8 sections total
- Return ONLY JSON"""

        user_prompt = f"""User Profile:
{json.dumps(user_profile.dict(exclude_none=True), indent=2)}

Conversation:
{convo_text}

Generate the checklist:"""

        try:
            response = self.llm.invoke(f"{system_prompt}\n\n{user_prompt}")
            content = response.content if hasattr(response, 'content') else str(response)
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                sections = data.get("sections", [])
                
                for section in sections:
                    details = section.get("details", [])
                    if details:
                        done_count = sum(1 for d in details if d.get("status") == "done")
                        section["completion"] = int((done_count / len(details)) * 100)
                    else:
                        section["completion"] = 0
                    
                    section["status"] = "done" if section["completion"] == 100 else "pending"
                
                return sections
        except Exception as e:
            print(f"⚠️ Checklist generation failed: {e}")
        
        return []

# ==========================================
# 6. Orchestrator Agent
# ==========================================
class OrchestratorAgent:
    """Central coordinator using LLM-enhanced decision making"""
    
    def __init__(self, llm, intake, rag, tool, visual):
        self.llm = llm
        self.intake = intake
        self.rag = rag
        self.tool = tool
        self.visual = visual
    
    def route(self, user_input: str, user_profile: UserProfile) -> str:
        user_lower = user_input.lower().strip()
        
        # Simple greetings
        if user_lower in ['hi', 'hello', 'hey', 'start', 'begin', 'help']:
            completeness = self.intake.check_completeness(user_profile)
            if not completeness['complete']:
                return self.intake.get_questionnaire()
            return self.intake.get_smart_followup(user_profile)
        
        # LLM decision for complex queries
        return self._llm_decide_and_act(user_input, user_profile)
    
    def _llm_decide_and_act(self, user_input: str, user_profile: UserProfile) -> str:
        decision_prompt = f"""Analyze what the user needs.

User Profile: {user_profile.dict(exclude_none=True)}
Question: {user_input}

Decide: SEARCH, CALCULATE, BOTH, or DIRECT
Respond with ONE WORD only."""

        try:
            decision = self.llm.invoke(decision_prompt)
            action = decision.content.strip().upper()
            print(f"🤖 LLM Decision: {action}")
            
            if action == "CALCULATE" and user_profile.income:
                return self.tool.calculate_tax(user_profile.income)
            elif action == "SEARCH":
                context = self.rag.search(user_input)
                return self._synthesize(user_input, user_profile, context)
            elif action == "BOTH":
                context = self.rag.search(user_input)
                tax_info = self.tool.calculate_tax(user_profile.income) if user_profile.income else ""
                return self._synthesize(user_input, user_profile, context, tax_info)
            else:
                return self.rag.answer_with_context(user_input, user_profile)
        except Exception as e:
            return self.rag.answer_with_context(user_input, user_profile)
    
    def _synthesize(self, question: str, profile: UserProfile, context: str, tax_info: str = "") -> str:
        prompt = f"""Provide a helpful answer combining this information:

User Profile: {profile.dict(exclude_none=True)}
Question: {question}
IRS Documentation: {context}
{"Tax Calculation: " + tax_info if tax_info else ""}

Be friendly and clear! 😊"""
        
        response = self.llm.invoke(prompt)
        return response.content

# ==========================================
# 7. Main Orchestrator (External Interface)
# ==========================================
class TaxOrchestrator:
    """Main entry point managing all agents"""
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=api_key,
            temperature=0
        )
        
        print("🚀 Initializing Tax Assistant System...")
        
        self.intake_agent = IntakeAgent(self.llm)
        print("✅ Intake Agent ready")
        
        self.rag_agent = RAGAgent(self.llm)
        print("✅ RAG Agent ready")
        
        self.tool_agent = ToolAgent()
        print("✅ Tool Agent ready")
        
        # NEW: Visual Agent with RAG integration
        self.visual_agent = VisualAgent(self.llm, self.rag_agent)
        print("✅ Visual Agent ready (RAG-enhanced)")
        
        self.checklist_agent = ChecklistAgent(self.llm)
        print("✅ Checklist Agent ready")
        
        self.orchestrator = OrchestratorAgent(
            self.llm, self.intake_agent, self.rag_agent, 
            self.tool_agent, self.visual_agent
        )
        print("✅ Orchestrator ready")
        print("=" * 50)
    
    def run_orchestrator(self, user_input: str, user_profile: UserProfile = None) -> dict:
        if user_profile is None:
            user_profile = UserProfile()
        response = self.orchestrator.route(user_input, user_profile)
        return {"output": response}
    
    def run_intake(self, user_input: str) -> UserProfile:
        return self.intake_agent.extract_info(user_input)
    
    def generate_checklist(self, conversation_history: List[dict], user_profile: UserProfile = None) -> List[dict]:
        if user_profile is None:
            user_profile = UserProfile()
        return self.checklist_agent.generate_checklist(conversation_history, user_profile)
    
    # NEW: Visual generation methods
    def infer_visual_topic(self, messages: List[dict], user_profile: UserProfile) -> str:
        return self.visual_agent.infer_topic(messages, user_profile)
    
    def generate_visual_step(self, topic: str, user_profile: UserProfile) -> str:
        return self.visual_agent.generate_visual_snippet(topic, user_profile)
    
    def get_visual_snippets(self, topic: str) -> List[str]:
        return self.visual_agent.get_all_snippets(topic)
    
    def reset_visuals(self, topic: str = None):
        self.visual_agent.reset_topic(topic)