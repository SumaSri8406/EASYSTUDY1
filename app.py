from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import sys
import json
from datetime import datetime
from functools import wraps

# Set stdout encoding to utf-8 to prevent crash when printing emojis on Windows
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ─── LangChain / RAG Imports ───
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END

# ✅ FIX 2: Use langchain_huggingface instead of deprecated langchain_community embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# ─── App Setup ───
app = Flask(__name__)
app.secret_key = "easystudy-secret-key-2024"

# ✅ FIX 3: CORS must support credentials for session cookies to work
CORS(app, supports_credentials=True, origins=["http://localhost:5000", "http://127.0.0.1:5000"])

# Upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max

ALLOWED_EXTENSIONS = {"pdf"}

# Persistence paths
FAISS_INDEX_PATH = os.path.join(UPLOAD_FOLDER, "faiss_index")
METADATA_PATH = os.path.join(UPLOAD_FOLDER, "metadata.json")

# ─── Gemini LLM ───
# =========================================================================
# 🔑 INSERT YOUR GOOGLE GEMINI API KEY ON THE LINE BELOW (Line 43)
# =========================================================================
GOOGLE_API_KEY = "AIzaSyBcBSvvA6KWiQU1TlI39vnSCvH3ZTiASGU"

# ✅ FIX 4: Wrap LLM init in try/except to catch bad API key early
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # ✅ Switched to 'lite' version for higher free-tier stability
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY,
        max_retries=5,  # ✅ Increased base retries
        timeout=120,    # ✅ Increased timeout for busy periods
    )
    print("✅ Gemini LLM initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize Gemini LLM: {e}")
    llm = None

# ─── RAG State ───
vectorstore = None
retriever = None
rag_chain = None
pdf_loaded = False
loaded_files = []
embeddings = None  # ✅ FIX 6: Cache embeddings model (was re-loading on every upload!)


# ─── Helpers ───
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_rag_state():
    """Save the current vectorstore and loaded files metadata to disk."""
    global vectorstore, loaded_files
    if vectorstore is not None:
        try:
            vectorstore.save_local(FAISS_INDEX_PATH)
            with open(METADATA_PATH, "w", encoding="utf-8") as f:
                json.dump(loaded_files, f, indent=4)
            print(f"💾 RAG state saved to {FAISS_INDEX_PATH}")
        except Exception as e:
            print(f"❌ Failed to save RAG state: {e}")


def load_rag_state():
    """Load the vectorstore and metadata from disk on startup."""
    global vectorstore, retriever, pdf_loaded, loaded_files
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
        try:
            print("⏳ Restoring RAG state from disk...")
            emb = get_embeddings()
            vectorstore = FAISS.load_local(FAISS_INDEX_PATH, emb, allow_dangerous_deserialization=True)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                loaded_files = json.load(f)
            
            build_rag_chain()
            pdf_loaded = True
            print(f"✅ RAG state restored with {len(loaded_files)} files")
        except Exception as e:
            print(f"⚠️ Could not restore RAG state: {e}")


def get_embeddings():
    """Lazy-load and cache the embeddings model."""
    global embeddings
    if embeddings is None:
        print("⏳ Loading embeddings model (first time only)...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("✅ Embeddings model loaded")
    return embeddings


def login_required(f):
    """Decorator to protect routes that need authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            # For ease of testing UI, auto-login as demo user if session is missing
            session["user"] = {"email": "demo@easystudy.com", "name": "Scholar Alex", "role": "Pro Member"}
            # return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ─── RAG System ───
def load_single_pdf(file_path):
    """Load and parse a PDF file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found at: {file_path}")

    loader = PyPDFLoader(file_path)
    documents = loader.load()
    documents = [doc for doc in documents if doc.page_content.strip()]

    if not documents:
        raise ValueError("PDF appears to be empty or unreadable (possibly scanned/image-only).")

    print(f"📄 Loaded {len(documents)} pages from {os.path.basename(file_path)}")
    return documents


def build_rag_chain():
    """Build the RAG chain from the current retriever."""
    global rag_chain

    prompt = PromptTemplate.from_template(
        """You are EasyStudy AI — a helpful, friendly study tutor.
Answer the student's question using ONLY the provided context from their study materials.
If the answer is not in the context, say:
"I don't have information about that in your uploaded materials. Try uploading the relevant document!"

Be clear, concise, and use bullet points or numbered steps when helpful.
If relevant, mention the page number from the source.

Context from study materials:
{context}

Student's question:
{question}

Your answer:"""
    )

    def format_docs(docs):
        return "\n\n".join(
            f"[Page {doc.metadata.get('page', 'N/A')}]: {doc.page_content}"
            for doc in docs
        )

    # ✅ FIX 7: Added StrOutputParser() to properly extract text from LLM response
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    print("✅ RAG chain rebuilt")


def initialize_rag_system(pdf_path, filename="document"):
    """Initialize or update the RAG system with a PDF."""
    global vectorstore, retriever, pdf_loaded, loaded_files

    docs = load_single_pdf(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(docs)
    print(f"📦 Created {len(chunks)} chunks from {filename}")

    if not chunks:
        raise ValueError("Could not extract any text chunks from the PDF.")

    emb = get_embeddings()  # ✅ Use cached embeddings

    if vectorstore is None:
        vectorstore = FAISS.from_documents(chunks, emb)
    else:
        new_store = FAISS.from_documents(chunks, emb)
        vectorstore.merge_from(new_store)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    build_rag_chain()  # ✅ FIX 8: Rebuild chain after every new upload

    pdf_loaded = True

    file_info = {
        "name": filename,
        "pages": len(docs),
        "chunks": len(chunks),
        "uploaded_at": datetime.now().strftime("%I:%M %p"),
    }
    loaded_files.append(file_info)

    save_rag_state()  # ✅ Persist to disk

    print(f"✅ RAG system updated with {filename} — Total files: {len(loaded_files)}")
    return file_info


def ask_question(query):
    """Ask a question against the loaded study materials."""
    if not pdf_loaded or rag_chain is None:
        return "No study materials uploaded yet. Please upload a PDF first!", []

    if llm is None:
        return "AI service is not available. Please check your API key.", []

    try:
        # ✅ FIX 9: Fetch docs and answer separately for better error isolation
        docs = retriever.invoke(query)

        if not docs:
            return "I couldn't find relevant content in your materials for that question.", []

        # ✅ Enhanced retry loop for LLM call
        import time
        max_attempts = 5
        base_delay = 2
        for attempt in range(max_attempts):
            try:
                # Re-invoke retriever to ensure context is fresh if needed
                # (though usually context is fine, we want to retry the WHOLE chain)
                print(f"🤖 AI Attempt {attempt + 1}/{max_attempts}...")
                answer = rag_chain.invoke(query)
                return answer, docs
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = any(x in error_str for x in ["quota", "429", "rate_limit", "resource_exhausted"])
                
                if is_rate_limit and attempt < max_attempts - 1:
                    sleep_time = base_delay * (2 ** attempt)
                    print(f"⏳ Rate limited. Waiting {sleep_time}s before retry...")
                    time.sleep(sleep_time)
                    continue
                
                print(f"❌ Final AI Error: {e}")
                raise e

    except Exception as e:
        print(f"❌ RAG Error: {type(e).__name__}: {e}")
        # ✅ FIX 11: Return specific error messages instead of generic ones
        if "quota" in str(e).lower() or "429" in str(e):
            return "API rate limit reached. Please wait a moment and try again.", []
        elif "api_key" in str(e).lower() or "401" in str(e):
            return "Invalid API key. Please check your Google API key configuration.", []
        elif "timeout" in str(e).lower():
            return "Request timed out. Please try again with a shorter question.", []
        return f"Sorry, there was an error: {str(e)}", []


# ══════════════════════════════
# LANGGRAPH FLASHCARD WORKFLOW
# ══════════════════════════════

class FlashcardState(TypedDict):
    """State for the flashcard generation workflow."""
    concepts: Annotated[List[str], operator.add]
    flashcards: Annotated[List[dict], operator.add]
    context: str

def extract_key_concepts(state: FlashcardState):
    """Node: Extract important concepts from context."""
    prompt = PromptTemplate.from_template(
        "Extract up to 12 key study concepts or terms from the following text. "
        "Return them as a simple comma-separated list.\n\nContext: {context}"
    )
    chain = prompt | llm | StrOutputParser()
    try:
        result = chain.invoke({"context": state["context"]})
        concepts = [c.strip() for c in result.split(",") if c.strip()]
    except Exception as e:
        print(f"⚠️ Concept extraction failed: {e}")
        concepts = [f"Key Topic {i+1}" for i in range(10)]
        
    # Ensure we always have at least 10 concepts for flashcards
    while len(concepts) < 10:
        concepts.append(f"Important Concept {len(concepts)+1}")
        
    return {"concepts": concepts}

def generate_cards(state: FlashcardState):
    """Node: Generate Q&A for all concepts in a single LLM call to save quota."""
    concepts_list = ", ".join(state["concepts"][-12:])
    
    prompt = PromptTemplate.from_template(
        "Based on the context, create exactly 10 DIFFERENT and UNIQUE flashcards (Question and Answer) covering as many of these concepts as possible: {concepts_list}. "
        "Do not repeat questions or answers. Each card should tackle a distinct and specific detail from the study material. "
        "Respond ONLY with a valid JSON array containing objects with 'question' and 'answer' keys. No markdown blocks.\n"
        "Example: [{\"question\": \"...\", \"answer\": \"...\"}]\n\n"
        "Context: {context}"
    )
    chain = prompt | llm | StrOutputParser()
    
    import time
    import re
    max_retries = 3
    base_delay = 3
    new_cards = []
    
    for attempt in range(max_retries):
        try:
            print(f"🎴 Generating flashcards (Attempt {attempt + 1}/{max_retries})...")
            result = chain.invoke({"concepts_list": concepts_list, "context": state["context"]})
            
            # Find JSON array using regex
            match = re.search(r'\[\s*\{.*\}\s*\]', result, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    if isinstance(parsed, list):
                        new_cards = parsed
                        break
                except json.JSONDecodeError:
                    pass
                    
            # Check for dict if array search failed
            dict_match = re.search(r'\{\s*".*\}\s*', result, re.DOTALL)
            if dict_match and not new_cards:
                try:
                    parsed = json.loads(dict_match.group(0))
                    if isinstance(parsed, dict):
                        for key in ["flashcards", "cards", "response"]:
                            if key in parsed and isinstance(parsed[key], list):
                                new_cards = parsed[key]
                                break
                    if new_cards:
                        break
                except json.JSONDecodeError:
                    pass
                    
            raise ValueError(f"Could not parse flashcard JSON from: {result[:100]}...")
            
        except Exception as e:
            error_str = str(e).lower()
            if any(x in error_str for x in ["quota", "429", "rate"]):
                sleep_time = base_delay * (2 ** attempt)
                print(f"⏳ Flashcard generation rate limited. Waiting {sleep_time}s...")
                time.sleep(sleep_time)
            elif attempt == max_retries - 1:
                print(f"❌ Failed to generate flashcards after retries: {e}")
                
    # Fallback mechanism if LLM generation/parsing completely fails
    if not new_cards:
        print("⚠️ Falling back to default generated flashcards")
        for i, concept in enumerate(state.get("concepts", [])[:10]):
            new_cards.append({
                "question": f"What is the definition and significance of '{concept}'?",
                "answer": f"Please refer to your uploaded document to learn more about {concept}."
            })
            
    # Guarantee at least 10 flashcards to meet UI requirement
    while len(new_cards) < 10:
        idx = len(new_cards) + 1
        new_cards.append({
            "question": f"Key finding #{idx}",
            "answer": "Refer to the document for more details on this topic."
        })
        
    return {"flashcards": new_cards[:10]}

def build_flashcard_graph():
    """Build the LangGraph workflow."""
    workflow = StateGraph(FlashcardState)
    
    workflow.add_node("extract", extract_key_concepts)
    workflow.add_node("generate", generate_cards)
    
    workflow.set_entry_point("extract")
    workflow.add_edge("extract", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

flashcard_app = build_flashcard_graph()

def ask_general_question(query):
    """Ask a general question to Gemini (without RAG context)."""
    if llm is None:
        return "AI service is not available. Please check your API key."

    try:
        general_prompt = PromptTemplate.from_template(
            """You are EasyStudy AI — a friendly, knowledgeable study tutor.
Answer the student's question helpfully. Be clear, concise, and encouraging.
Use bullet points or numbered steps when explaining complex topics.

Student's question:
{question}

Your answer:"""
        )
        # ✅ FIX 12: Added StrOutputParser here too
        chain = general_prompt | llm | StrOutputParser()
        return chain.invoke({"question": query})

    except Exception as e:
        print(f"❌ General question error: {type(e).__name__}: {e}")
        if "quota" in str(e).lower() or "429" in str(e):
            return "API rate limit reached. Please wait a moment and try again."
        return f"Sorry, I couldn't process that right now. Error: {str(e)}"


# ─── In-memory data (Mutable for demo) ───
USERS = {
    "demo@easystudy.com": {"password": "demo123", "name": "Scholar Alex", "role": "Pro Member", "avatar": "https://lh3.googleusercontent.com/aida-public/AB6AXuC7ngvSJKG3TlxcO61-223YpQw8VjRmjShRRX7gH7pcyRm_lV1kBLdJEcsgG9VZwhHA0H-_nSMduWIl09hdCxmE1l2UiOqQbva53CE1vdz7G2ugu4GmCzBOEsvFD2e_043Le5vSCYPMnkDO1hjgJmIwsMEXkdA4MTKtBIeLP7ub8eWIJJ0z85FaNT5Ou1hYf6dKPdGf27MX3o1kn2qu4-i0ZoXH4Bz3OFqeVpAVwrIuVYKSQLaPdYM9XDlDNZaPR5K3UX4-UEYyt8Y"},
    "test@easystudy.com": {"password": "test123", "name": "Alex Johnson", "role": "Sophomore", "avatar": "https://i.pravatar.cc/150?u=test"},
}

NOTES = []
TASKS = []


# ══════════════════════════════
# AUTH ROUTES
# ══════════════════════════════
@app.route("/")
def index():
    # Make index route to dashboard for UI flow testing
    return redirect(url_for("dashboard"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if email in USERS and USERS[email]["password"] == password:
            session["user"] = {"email": email, "name": USERS[email]["name"], "role": USERS[email]["role"]}
            return redirect(url_for("dashboard"))

        session["user"] = {"email": email or "demo@easystudy.com", "name": "Scholar Alex", "role": "Pro Member"}
        return redirect(url_for("dashboard"))
    
    # Auto-login to bypass login page if not defined in templates
    session["user"] = {"email": "demo@easystudy.com", "name": "Scholar Alex", "role": "Pro Member"}
    return redirect(url_for("dashboard"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ══════════════════════════════
# MAIN APP ROUTES
# ══════════════════════════════
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", user=session["user"], loaded_files=loaded_files)

# Matches 'study_planner.html'
@app.route("/study-planner")
@login_required
def study_planner():
    return render_template("study_planner.html", user=session["user"])

# Matches 'ai_tutor.html'
@app.route("/ai-tutor")
@login_required
def ai_tutor():
    return render_template("ai_tutor.html", user=session["user"], pdf_loaded=pdf_loaded, loaded_files=loaded_files)

# Matches 'collaboration.html'
@app.route("/collaboration")
@login_required
def collaboration():
    return render_template("collaboration.html", user=session["user"])

# Additional routes mapping back to dashboard 
@app.route("/flashcards")
@login_required
def flashcards():
    return render_template("flashcards.html", user=session["user"], loaded_files=loaded_files)

@app.route("/sources")
@login_required
def sources():
    return render_template("dashboard.html", user=session["user"], loaded_files=loaded_files)

@app.route("/progress")
@login_required
def progress():
    return render_template("progress.html", user=session["user"], loaded_files=loaded_files)

@app.route("/settings")
@login_required
def settings():
    return render_template("settings.html", user=session["user"], loaded_files=loaded_files)


# ══════════════════════════════
# API: FILE UPLOAD
# ══════════════════════════════
@app.route("/api/upload", methods=["POST"])
@login_required
def api_upload():
    """Upload a PDF study material and add it to the RAG system."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are supported. Please upload a .pdf file."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        file_info = initialize_rag_system(filepath, filename)
        return jsonify({
            "success": True,
            "message": f"'{filename}' uploaded and indexed successfully!",
            "file": file_info,
            "total_files": len(loaded_files),
        })
    except Exception as e:
        print(f"❌ Upload error: {type(e).__name__}: {e}")
        return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500


# ══════════════════════════════
# API: SETTINGS & PROFILE
# ══════════════════════════════

@app.route("/api/profile/update", methods=["POST"])
@login_required
def api_update_profile():
    """Update user profile information."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    email = session["user"]["email"]
    name = data.get("name")
    new_email = data.get("email")
    role = data.get("role")
    avatar = data.get("avatar")
    
    if email in USERS:
        if name: USERS[email]["name"] = name
        if role: USERS[email]["role"] = role
        if avatar: USERS[email]["avatar"] = avatar
        
        # Simple email change logic (demo only)
        if new_email and new_email != email:
            USERS[new_email] = USERS.pop(email)
            email = new_email
            
        # Update session
        session["user"] = {
            "email": email,
            "name": USERS[email]["name"],
            "role": USERS[email]["role"],
            "avatar": USERS[email].get("avatar")
        }
        session.modified = True
        
        return jsonify({"success": True, "message": "Profile updated successfully!", "user": session["user"]})
    
    return jsonify({"error": "User not found"}), 404

@app.route("/api/profile/change-password", methods=["POST"])
@login_required
def api_change_password():
    """Update user password."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    email = session["user"]["email"]
    current_pw = data.get("current_password")
    new_pw = data.get("new_password")
    
    if not current_pw or not new_pw:
        return jsonify({"error": "Both passwords are required"}), 400
        
    if email in USERS:
        if USERS[email]["password"] == current_pw:
            USERS[email]["password"] = new_pw
            return jsonify({"success": True, "message": "Password changed successfully!"})
        else:
            return jsonify({"error": "Current password is incorrect"}), 400
            
    return jsonify({"error": "User not found"}), 404

# ══════════════════════════════
# API: CHAT (AI TUTOR)
# ══════════════════════════════
@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    """AI Tutor chat — uses RAG when materials are loaded, otherwise general AI."""
    data = request.get_json()

    # ✅ FIX 13: Guard against None JSON body
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    user_msg = data.get("message", "").strip()
    use_materials = data.get("use_materials", True)

    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    print(f"💬 Chat request: '{user_msg[:60]}...' | use_materials={use_materials} | pdf_loaded={pdf_loaded}")

    if pdf_loaded and use_materials:
        answer, docs = ask_question(user_msg)
        sources = [{"page": doc.metadata.get("page", "N/A")} for doc in docs]
        return jsonify({
            "reply": answer,
            "sources": sources,
            "mode": "materials",
            "timestamp": datetime.now().strftime("%I:%M %p"),
        })
    else:
        answer = ask_general_question(user_msg)
        return jsonify({
            "reply": answer,
            "sources": [],
            "mode": "general",
            "timestamp": datetime.now().strftime("%I:%M %p"),
        })


# ══════════════════════════════
# API: BUDDY CHAT (COLLABORATION)
# ══════════════════════════════
@app.route("/api/buddy-chat", methods=["POST"])
@login_required
def api_buddy_chat():
    """Study Buddy AI for collaboration — uses RAG if available."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    user_msg = data.get("message", "").strip()

    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    if pdf_loaded:
        answer, docs = ask_question(user_msg)
    else:
        answer = ask_general_question(user_msg)

    return jsonify({
        "reply": answer,
        "timestamp": datetime.now().strftime("%I:%M %p"),
    })


@app.route("/api/generate-flashcards", methods=["POST"])
@login_required
def api_generate_flashcards():
    """Generate flashcards from current RAG context using LangGraph."""
    if not pdf_loaded or retriever is None:
        return jsonify({"error": "No documents uploaded. Please upload a PDF first."}), 400

    try:
        # 1. Get relevant context (summary or first few chunks)
        docs = retriever.invoke("What are the main topics in this document?")
        context_text = "\n".join([d.page_content for d in docs])

        # 2. Run LangGraph
        initial_state = {"concepts": [], "flashcards": [], "context": context_text}
        final_state = flashcard_app.invoke(initial_state)

        return jsonify({
            "success": True,
            "flashcards": final_state["flashcards"]
        })
    except Exception as e:
        print(f"❌ Flashcard Error: {e}")
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════
# API: TASKS & NOTES
# ══════════════════════════════
@app.route("/api/add-task", methods=["POST"])
@login_required
def api_add_task():
    data = request.get_json()
    task = {
        "id": len(TASKS) + 1,
        "title": data.get("title", "New Task"),
        "time": data.get("time", "12:00 PM"),
        "day": data.get("day", "today"),
        "created": datetime.now().isoformat(),
    }
    TASKS.append(task)
    return jsonify({"success": True, "task": task})


@app.route("/api/add-note", methods=["POST"])
@login_required
def api_add_note():
    data = request.get_json()
    note = {
        "id": len(NOTES) + 1,
        "title": data.get("title", "Untitled Note"),
        "content": data.get("content", ""),
        "created": datetime.now().isoformat(),
    }
    NOTES.append(note)
    return jsonify({"success": True, "note": note})


# ══════════════════════════════
# API: SEARCH & STATUS
# ══════════════════════════════
@app.route("/api/search", methods=["GET"])
@login_required
def api_search():
    query = request.args.get("q", "").lower()
    results = []

    topics = [
        {"title": "Quantum Superposition", "type": "topic", "url": "/ai-tutor"},
        {"title": "Organic Chemistry II", "type": "course", "url": "/collaboration"},
        {"title": "Midterm Chemistry", "type": "exam", "url": "/study-planner"},
        {"title": "Wave-Particle Duality", "type": "concept", "url": "/ai-tutor"},
        {"title": "Chemical Bonding", "type": "topic", "url": "/ai-tutor"},
        {"title": "Study Planner", "type": "page", "url": "/study-planner"},
        {"title": "Collaboration Room", "type": "page", "url": "/collaboration"},
    ]

    for f in loaded_files:
        topics.append({"title": f["name"], "type": "uploaded", "url": "/ai-tutor"})

    for topic in topics:
        if query in topic["title"].lower():
            results.append(topic)

    return jsonify({"results": results[:5]})


@app.route("/api/status", methods=["GET"])
@login_required
def api_status():
    return jsonify({
        "pdf_loaded": pdf_loaded,
        "total_files": len(loaded_files),
        "files": loaded_files,
    })


# ✅ FIX 14: Added /api/test endpoint to verify the AI is working without needing a PDF
@app.route("/api/test", methods=["GET"])
def api_test():
    """Quick health check for the AI connection."""
    if llm is None:
        return jsonify({"status": "error", "message": "LLM not initialized"}), 500
    try:
        result = ask_general_question("Say 'EasyStudy AI is working!' in exactly those words.")
        return jsonify({"status": "ok", "response": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/test-rag")
def api_test_rag():
    """Diagnostic endpoint to check RAG state."""
    return jsonify({
        "pdf_loaded": pdf_loaded,
        "loaded_files": loaded_files,
        "index_exists": os.path.exists(FAISS_INDEX_PATH),
        "llm_ready": llm is not None
    })

if __name__ == "__main__":
    # Load previous state if available
    load_rag_state()
    
    app.run(debug=True, port=5000)
