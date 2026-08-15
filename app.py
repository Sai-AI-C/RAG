import os
import json
import time
import uuid
import streamlit as st
import chromadb
import ollama
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

# 1. PAGE CONFIGURATION & STYLING
st.set_page_config(
    page_title="OmniDoc AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Base dark theme */
    .stApp { background-color: #0e1117; }

    /* Sidebar subject buttons */
    div[data-testid="stSidebarContent"] .stButton button {
        text-align: left;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 0.88rem;
        font-weight: 500;
        transition: background 0.2s;
    }
    div[data-testid="stSidebarContent"] .stButton button:hover {
        background-color: #4f46e5 !important;
        color: white !important;
        border-color: #4f46e5 !important;
    }

    /* Active subject header banner */
    .subj-banner {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 10px;
        margin-bottom: 18px;
        font-weight: 600;
        font-size: 1.05rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .subj-banner .subj-tag {
        font-size: 0.78rem;
        background: rgba(255,255,255,0.22);
        padding: 2px 10px;
        border-radius: 20px;
        font-weight: 500;
        margin-left: auto;
    }
    /* Sample question chips */
    .sample-chip {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 20px;
        padding: 8px 16px;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s;
        color: #cbd5e1;
        display: inline-block;
        margin: 4px;
    }
    .sample-chip:hover {
        border-color: #6366f1;
        color: white;
        background: #1d1f3a;
    }
    /* Subject category label in sidebar */
    .sidebar-cat {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        padding: 6px 0 2px 4px;
    }
    /* Active subject button */
    .active-subj-btn button {
        background-color: #4f46e5 !important;
        color: white !important;
        border-color: #4f46e5 !important;
        font-weight: 700 !important;
    }
    /* Mobile responsive */
    @media (max-width: 768px) {
        .subj-banner { flex-direction: column; align-items: flex-start; gap: 6px; }
        .subj-banner .subj-tag { margin-left: 0; }
        .stButton button { min-height: 40px; }
    }
</style>
""", unsafe_allow_html=True)

CHAT_SESSIONS_DIR="./chat_history_sessions"
os.makedirs(CHAT_SESSIONS_DIR,exist_ok=True)

# 2. SUBJECT METADATA & SAMPLE QUESTIONS
SUBJECT_METADATA = {
    "AI":{"title":"Artificial Intelligence","category":"AI & Data Science","icon":"🤖","type":"Notes"},
    "AI-NLP Lab":{"title":"AI & NLP Lab","category":"AI & Data Science","icon":"🧪","type":"Lab"},
    "ML notes":{"title":"Machine Learning","category":"AI & Data Science","icon":"🧠","type":"Notes"},
    "ML Lab":{"title":"Machine Learning Lab","category":"AI & Data Science","icon":"🔬","type":"Lab"},
    "NLP":{"title":"Natural Language Processing","category":"AI & Data Science","icon":"💬","type":"Notes"},
    "Neural network and deep learning":{"title":"Neural Networks & Deep Learning","category":"AI & Data Science","icon":"🕸️","type":"Notes"},
    "Reinforcement Learning":{"title":"Reinforcement Learning","category":"AI & Data Science","icon":"🎮","type":"Notes"},
    "CN Notes":{"title":"Computer Networks","category":"Networks & Security","icon":"📡","type":"Notes"},
    "CN Lab":{"title":"Computer Networks Lab","category":"Networks & Security","icon":"🔌","type":"Lab"},
    "CNS":{"title":"Cryptography & Network Security","category":"Networks & Security","icon":"🔐","type":"Notes"},
    "CNS Lab":{"title":"CNS Lab","category":"Networks & Security","icon":"🛡️","type":"Lab"},
    "Cloud Computing":{"title":"Cloud Computing","category":"Networks & Security","icon":"☁️","type":"Notes"},
    "SNA":{"title":"Social Network Analysis","category":"Networks & Security","icon":"🌐","type":"Notes"},
    "CD Notes":{"title":"Compiler Design","category":"Core CS & Systems","icon":"⚙️","type":"Notes"},
    "DAA Notes":{"title":"Design & Analysis of Algorithms","category":"Core CS & Systems","icon":"🧮","type":"Notes"},
    "DBMS":{"title":"Database Management Systems","category":"Core CS & Systems","icon":"🗄️","type":"Notes"},
    "DBMS Lab":{"title":"DBMS Lab","category":"Core CS & Systems","icon":"💾","type":"Lab"},
    "OS":{"title":"Operating Systems","category":"Core CS & Systems","icon":"💻","type":"Notes"},
    "COA":{"title":"Computer Organization & Architecture","category":"Core CS & Systems","icon":"🏗️","type":"Notes"},
    "FLAT":{"title":"Formal Languages & Automata Theory","category":"Core CS & Systems","icon":"🔢","type":"Notes"},
    "Software Engineering":{"title":"Software Engineering","category":"Core CS & Systems","icon":"🛠️","type":"Notes"},
    "Devops":{"title":"DevOps","category":"Core CS & Systems","icon":"🚀","type":"Notes"},
    "Devops lab":{"title":"DevOps Lab","category":"Core CS & Systems","icon":"🐳","type":"Lab"},
    "Data structure":{"title":"Data Structures","category":"Core CS & Systems","icon":"🌳","type":"Notes"},
    "BEFA":{"title":"Business Economics & Financial Analysis","category":"Management & Electives","icon":"📊","type":"Notes"},
    "DM":{"title":"Discrete Mathematics / Data Mining","category":"Management & Electives","icon":"📐","type":"Notes"},
    "DPPM":{"title":"Data Preparation & Pattern Mining","category":"Management & Electives","icon":"⛏️","type":"Notes"},
    "Java":{"title":"Java Programming","category":"Management & Electives","icon":"☕","type":"Notes"},
    "Java Lab":{"title":"Java Lab","category":"Management & Electives","icon":"🍵","type":"Lab"},
    "MSF":{"title":"Management Science & Finance","category":"Management & Electives","icon":"📈","type":"Notes"},
    "Organizational Behaviour":{"title":"Organizational Behaviour","category":"Management & Electives","icon":"🏢","type":"Notes"},
    "POE":{"title":"Principles of Economics","category":"Management & Electives","icon":"💰","type":"Notes"},
    "PP":{"title":"Python Programming","category":"Management & Electives","icon":"🐍","type":"Notes"},
    "STM Notes":{"title":"Software Testing Methodologies","category":"Management & Electives","icon":"🧪","type":"Notes"},
    "Semantic Web":{"title":"Semantic Web","category":"Management & Electives","icon":"🕸️","type":"Notes"},
    "Total Quality Management":{"title":"Total Quality Management","category":"Management & Electives","icon":"🎯","type":"Notes"},
    "WP Notes":{"title":"Web Programming","category":"Management & Electives","icon":"🌐","type":"Notes"},
    "ACS Lab":{"title":"Advanced Communication Systems Lab","category":"Management & Electives","icon":"📡","type":"Lab"},
}

# Per-subject sample questions shown before first message
SUBJECT_SAMPLE_QUESTIONS={
    "AI":               ["Give AI lab list of experiments",       "What is Turing Test?",                  "Explain A* search algorithm",           "What is Heuristic Search?"],
    "AI-NLP Lab":       ["List all NLP lab experiments",          "What is tokenization in NLP?",          "Explain Named Entity Recognition",      "Give AI NLP lab programs list"],
    "ML notes":         ["How many types of Machine Learning?",   "Explain supervised vs unsupervised",    "What is overfitting in ML?",            "Explain Decision Tree algorithm"],
    "ML Lab":           ["List all ML lab experiments",           "How to implement KNN algorithm?",       "Explain SVM with example",              "What is Naive Bayes classifier?"],
    "NLP":              ["What is NLP? Define it",                "Explain parsing in NLP",                "What is stemming and lemmatization?",   "NLP applications in real life"],
    "Neural network and deep learning": ["What is Deep Learning?","Explain Backpropagation algorithm",    "What is CNN vs RNN?",                   "Define activation functions"],
    "Reinforcement Learning": ["What is RL? Define it",           "Explain Q-learning algorithm",          "What is Markov Decision Process?",      "RL vs supervised learning"],

    "CN Notes":         ["What is Computer Networks (CN)?",       "Explain OSI model 7 layers",            "What is TCP vs UDP?",                   "Explain IP addressing and subnetting"],
    "CN Lab":           ["List all CN lab experiments",           "What is socket programming?",           "Explain ping and traceroute",           "TCP vs UDP lab experiment"],
    "CNS":              ["What is Cryptography?",                 "Explain RSA algorithm",                 "What is Digital Signature?",            "Explain AES encryption"],
    "CNS Lab":          ["List all CNS lab experiments",          "Implement Caesar cipher",               "What is DES algorithm?",                "CNS lab programs list"],
    "Cloud Computing":  ["What is Cloud Computing?",              "Explain SaaS PaaS IaaS",                "What is virtualization?",               "Types of cloud deployment models"],
    "SNA":              ["What is Social Network Analysis?",      "Explain centrality measures",           "What is graph theory in SNA?",          "Network clustering algorithms"],

    "CD Notes":         ["What are phases of Compiler Design?",   "Explain lexical analysis",              "What is syntax analysis / parsing?",    "Explain code optimization"],
    "DAA Notes":        ["What is DAA? Define DA",                "Explain time complexity Big O",         "What is Dynamic Programming?",          "Explain Greedy algorithms"],
    "DBMS":             ["What is DBMS?",                         "Explain normalization forms",           "What is SQL vs NoSQL?",                 "Explain ER diagram"],
    "DBMS Lab":         ["List all DBMS lab experiments",         "Write SQL queries for joins",           "Create database with DBMS lab",         "Explain triggers and procedures"],
    "OS":               ["What is Operating System?",             "Explain process scheduling algorithms", "What is deadlock and its prevention?",  "Explain memory management"],
    "COA":              ["What is Computer Organization?",        "Explain CPU architecture",              "What is pipelining in COA?",            "Explain memory hierarchy"],
    "FLAT":             ["What is Automata Theory?",              "Explain DFA vs NFA",                   "What is pushdown automaton?",           "Explain Turing Machine"],
    "Software Engineering": ["What is SDLC?",                     "Explain Agile methodology",             "What is software testing?",             "Explain UML diagrams"],
    "Devops":           ["What is DevOps?",                       "Explain CI/CD pipeline",                "What is Docker and Kubernetes?",        "DevOps tools overview"],
    "Devops lab":       ["List DevOps lab experiments",           "Setup Docker container",                "Implement Jenkins CI/CD",               "Git workflow in DevOps"],
    "Data structure":   ["What are linear data structures?",      "Explain trees and graphs",              "What is sorting algorithms?",           "Explain stack and queue"],

    "BEFA":             ["What is business economics?",           "Explain demand and supply",             "Financial ratio analysis",              "What is break-even analysis?"],
    "DM":               ["What is Data Mining?",                  "Explain association rules",             "What is clustering in DM?",             "Data mining algorithms"],
    "DPPM":             ["What is DPPM?",                         "Explain data preprocessing",            "What is pattern mining?",               "Feature engineering techniques"],
    "Java":             ["What is OOP in Java?",                  "Explain Java inheritance",              "What is Exception Handling in Java?",   "Java collections framework"],
    "Java Lab":         ["List all Java lab programs",            "Implement Java thread program",         "Java file handling program",            "Write Java socket program"],
    "MSF":              ["What is Management Science?",           "Explain linear programming",            "What is operations research?",          "Financial management basics"],
    "Organizational Behaviour": ["What is OB?",                   "Explain motivation theories",           "What is organizational culture?",       "Leadership styles in OB"],
    "POE":              ["What is Economics?",                    "Explain microeconomics vs macroeconomics","What is GDP?",                        "Types of market structures"],
    "PP":               ["What is Python?",                       "Python data types and variables",       "Explain list comprehension",            "Python OOP concepts"],
    "STM Notes":        ["What is software testing?",             "Black box vs white box testing",        "What is unit testing?",                 "Explain test cases and test plans"],
    "Semantic Web":     ["What is Semantic Web?",                 "Explain RDF and OWL",                   "What is ontology?",                     "SPARQL query language"],
    "Total Quality Management": ["What is TQM?",                  "Explain Six Sigma",                     "What is ISO standards?",                "TQM tools and techniques"],
    "WP Notes":         ["What is HTML and CSS?",                 "Explain JavaScript basics",             "What is responsive web design?",        "Web frameworks overview"],
    "ACS Lab":          ["What is ACS Lab?",                      "List all ACS lab experiments",          "ACS lab record notes overview",         "Communication systems basics"],

    "All Subjects":     ["What is Computer Networks?",            "Give AI lab list of experiments",       "Explain normalization in DBMS",         "What are types of Machine Learning?"],
}

ABBREVIATION_MAP = {
    "cn": "Computer Networks", "cns": "Cryptography and Network Security",
    "cd": "Compiler Design", "ai": "Artificial Intelligence",
    "daa": "Design and Analysis of Algorithms", "dbms": "Database Management Systems",
    "os": "Operating Systems", "ml": "Machine Learning",
    "nndl": "Neural Networks and Deep Learning", "nlp": "Natural Language Processing",
    "flat": "Formal Languages and Automata Theory", "devops": "DevOps Practices and Tools",
    "se": "Software Engineering", "stm": "Software Testing Methodologies",
    "dppm": "Data Preparation and Pattern Mining",
    "da": "Design Analysis of Algorithms", "acs": "Advanced Communication Systems"
}


def expand_query(query: str, active_subject: str = "All Subjects") -> str:
    cleaned = query.strip().lower()
    if active_subject in ["CN Notes", "CN Lab"] and cleaned in ["cn", "define cn", "what is cn"]:
        return "Computer Networks architecture OSI TCP/IP model layers protocols definitions"
    if active_subject in ["AI", "AI-NLP Lab"] and cleaned in ["ai", "define ai", "what is ai"]:
        return "Artificial Intelligence definitions computational models agents turing test"
    if active_subject in ["ML notes", "ML Lab"] and cleaned in ["ml", "define ml", "what is ml", "how many types ml", "types ml"]:
        return "Machine Learning types supervised unsupervised reinforcement learning classification regression"
    if active_subject in ["DAA Notes"] and cleaned in ["da", "daa", "define da", "what is da"]:
        return "Design and Analysis of Algorithms asymptotic notations time complexity"
    if active_subject in ["ACS Lab"] and cleaned in ["acs", "define acs", "what is acs"]:
        return "Advanced Communication Systems lab experiments record notes"
    if cleaned in ABBREVIATION_MAP:
        return f"{query} ({ABBREVIATION_MAP[cleaned]})"
    return query

# 3. PERSISTENT SESSION STORAGE (PER-SUBJECT)
def _session_path(session_id: str) -> str:
    return os.path.join(CHAT_SESSIONS_DIR, f"{session_id}.json")

def get_all_sessions():
    sessions = []
    for fname in os.listdir(CHAT_SESSIONS_DIR):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(CHAT_SESSIONS_DIR, fname), "r", encoding="utf-8") as f:
                    sessions.append(json.load(f))
            except Exception:
                pass
    sessions.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return sessions

def load_session(session_id: str):
    p = _session_path(session_id)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None

def save_session(session_id: str, title: str, messages: list, subject: str = "All Subjects"):
    if not messages:
        return
    data = {
        "session_id": session_id,
        "title": title[:30] + ("..." if len(title) > 30 else ""),
        "timestamp": time.time(),
        "messages": messages,
        "subject": subject,
    }
    with open(_session_path(session_id), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def delete_session(session_id: str):
    p = _session_path(session_id)
    if os.path.exists(p):
        os.remove(p)

# 4. VECTOR STORE & MODELS
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def get_vector_store():
    persist_dir = "./pdf_db/chromadb"
    if not os.path.exists(persist_dir):
        st.error(f"Vector Database not found at '{persist_dir}'. Please run Load.py first.")
        st.stop()
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(name="Document__C")


@st.cache_data(show_spinner=False)
def get_subject_counts():
    collection = get_vector_store()
    try:
        results = collection.get(include=["metadatas"])
        counts = {}
        for m in results.get("metadatas", []):
            if m and "subject" in m:
                counts[m["subject"]] = counts.get(m["subject"], 0) + 1
        return counts
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def get_local_ollama_models():
    try:
        models_list = ollama.list().get("models", [])
        names = [m.get("name") or m.get("model") for m in models_list if m]
        if names:
            return names
    except Exception:
        pass
    return ["qwen2.5:7b", "llama3.2:latest", "llama3.1:8b", "mistral:7b"]

# 5. RAG RETRIEVAL WITH SUBJECT FILTERING
def get_related_subjects(subject: str) -> list:
    if not subject or subject == "All Subjects":
        return []
    related = [subject]
    if "Notes" in subject:
        related.append(subject.replace("Notes", "Lab").strip())
    elif "Lab" in subject:
        related.append(subject.replace("Lab", "Notes").strip())
    extra = {
        "AI": "AI-NLP Lab", "AI-NLP Lab": "AI",
        "DBMS": "DBMS Lab", "DBMS Lab": "DBMS",
        "Devops": "Devops lab", "Devops lab": "Devops",
        "Java": "Java Lab", "Java Lab": "Java",
    }
    if subject in extra:
        related.append(extra[subject])
    return list(set(related))


def retrieve_context(query: str, subject_filter: str = "All Subjects", k: int = 6) -> str:
    search_query = expand_query(query, active_subject=subject_filter)
    emb_model = load_embedding_model()
    collection = get_vector_store()
    query_embedding = emb_model.encode([search_query])[0].tolist()

    where_clause = None
    if subject_filter and subject_filter != "All Subjects":
        targets = get_related_subjects(subject_filter)
        if len(targets) == 1:
            where_clause = {"subject": targets[0]}
        elif len(targets) > 1:
            where_clause = {"subject": {"$in": targets}}

    results = collection.query(query_embeddings=[query_embedding], n_results=k, where=where_clause)
    docs = results.get("documents", [[]])[0]
    return "\n\n---\n\n".join(docs)

# 6. SESSION STATE INITIALIZATION
def _fresh_session_state(subject="All Subjects"):
    return {
        "active_session_id": uuid.uuid4().hex[:8],
        "messages": [],
        "session_title": "New Chat",
        "selected_subject": subject,
    }


if "active_session_id" not in st.session_state:
    st.session_state.update(_fresh_session_state())
    st.session_state.loaded_session_id = None

# Load session from disk only on session ID change
if st.session_state.get("loaded_session_id") != st.session_state.active_session_id:
    data = load_session(st.session_state.active_session_id)
    if data:
        st.session_state.messages = data.get("messages", [])
        st.session_state.session_title = data.get("title", "Chat Session")
        st.session_state.selected_subject = data.get("subject", "All Subjects")
    else:
        st.session_state.messages = []
        st.session_state.session_title = "New Chat"
        # Keep current selected_subject so switching subject → new chat keeps subject
    st.session_state.loaded_session_id = st.session_state.active_session_id


def switch_subject(new_subject: str):
    """Save current chat, then open/create a chat session for new_subject."""
    cur_subject = st.session_state.selected_subject

    # Save current chat if it has messages and subject is changing
    if st.session_state.messages and cur_subject != new_subject:
        save_session(
            st.session_state.active_session_id,
            st.session_state.session_title,
            st.session_state.messages,
            subject=cur_subject,
        )

    # Look for an existing unfinished session for this subject
    all_sessions = get_all_sessions()
    matching = [s for s in all_sessions if s.get("subject") == new_subject]

    if matching and len(matching[0].get("messages", [])) == 0:
        # Restore empty session
        s = matching[0]
        st.session_state.active_session_id = s["session_id"]
        st.session_state.messages = []
        st.session_state.session_title = "New Chat"
    else:
        # Start a fresh session for this subject
        new_id = uuid.uuid4().hex[:8]
        st.session_state.active_session_id = new_id
        st.session_state.messages = []
        st.session_state.session_title = "New Chat"

    st.session_state.selected_subject = new_subject
    st.session_state.loaded_session_id = st.session_state.active_session_id

# 7. SIDEBAR — SUBJECT LIST & CHAT HISTORY
SIDEBAR_CATEGORIES = {
    "🤖 AI & Data Science":    ["AI", "AI-NLP Lab", "ML notes", "ML Lab", "NLP", "Neural network and deep learning", "Reinforcement Learning"],
    "📡 Networks & Security":  ["CN Notes", "CN Lab", "CNS", "CNS Lab", "Cloud Computing", "SNA"],
    "💻 Core CS & Systems":    ["CD Notes", "DAA Notes", "DBMS", "DBMS Lab", "OS", "COA", "FLAT", "Software Engineering", "Devops", "Devops lab", "Data structure"],
    "📊 Management & Electives": ["BEFA", "DM", "DPPM", "Java", "Java Lab", "MSF", "Organizational Behaviour", "POE", "PP", "STM Notes", "Semantic Web", "Total Quality Management", "WP Notes", "ACS Lab"],
}

with st.sidebar:
    st.markdown("## ✨ OmniDoc AI")
    st.caption("Ask anything from your engineering notes")

    # New Chat button
    if st.button("➕ New Chat", use_container_width=True):
        cur_subj = st.session_state.selected_subject
        if st.session_state.messages:
            save_session(
                st.session_state.active_session_id,
                st.session_state.session_title,
                st.session_state.messages,
                subject=cur_subj,
            )
        new_id = uuid.uuid4().hex[:8]
        st.session_state.active_session_id = new_id
        st.session_state.loaded_session_id = new_id
        st.session_state.messages = []
        st.session_state.session_title = "New Chat"
        st.rerun()

    st.divider()

    # SUBJECT SELECTOR 
    st.markdown("**📚 Select Subject**")
    cur_subj = st.session_state.selected_subject

    # All Subjects button
    btn_type = "primary" if cur_subj == "All Subjects" else "secondary"
    if st.button("🌐 All Subjects", use_container_width=True, type=btn_type, key="btn_all"):
        if cur_subj != "All Subjects":
            switch_subject("All Subjects")
            st.rerun()

    # Per-category subject buttons
    for cat_name, subjects in SIDEBAR_CATEGORIES.items():
        st.markdown(f"<div class='sidebar-cat'>{cat_name}</div>", unsafe_allow_html=True)
        for subj in subjects:
            meta = SUBJECT_METADATA.get(subj, {})
            icon = meta.get("icon", "📖")
            title = meta.get("title", subj)
            stype = meta.get("type", "")
            label = f"{icon} {title}"
            if stype == "Lab":
                label += " 🔬"

            is_active = (cur_subj == subj)
            btn_style = "primary" if is_active else "secondary"
            if st.button(label, key=f"sbtn_{subj}", use_container_width=True, type=btn_style):
                if not is_active:
                    switch_subject(subj)
                    st.rerun()

    st.divider()

#CHAT HISTORY 
    st.markdown("**💬 Chat History**")
    saved = get_all_sessions()
    if saved:
        for s in saved:
            s_id = s["session_id"]
            s_title = s.get("title", "Chat")
            s_subj = s.get("subject", "All Subjects")
            is_active = (s_id == st.session_state.active_session_id)
            col_t, col_d = st.columns([0.82, 0.18])
            icon = "📌" if is_active else "💬"
            subj_meta = SUBJECT_METADATA.get(s_subj, {})
            subj_icon = subj_meta.get("icon", "📚")
            label = f"{icon} {s_title}\n{subj_icon} {s_subj}"
            if col_t.button(label, key=f"sess_{s_id}", use_container_width=True):
                if s_id != st.session_state.active_session_id:
                    st.session_state.active_session_id = s_id
                    st.session_state.messages = s.get("messages", [])
                    st.session_state.session_title = s_title
                    st.session_state.selected_subject = s_subj
                    st.session_state.loaded_session_id = s_id
                    st.rerun()
            if col_d.button("🗑️", key=f"del_{s_id}"):
                delete_session(s_id)
                if s_id == st.session_state.active_session_id:
                    new_id = uuid.uuid4().hex[:8]
                    st.session_state.active_session_id = new_id
                    st.session_state.loaded_session_id = new_id
                    st.session_state.messages = []
                    st.session_state.session_title = "New Chat"
                st.rerun()
    else:
        st.caption("No saved chats yet.")

    st.divider()

    # Model selector
    local_models = get_local_ollama_models()
    selected_model = st.selectbox("🧠 AI Model", local_models, index=0)
    collection = get_vector_store()
    # st.caption(f"📊 **{collection.count():,}** chunks · **38** subjects")

# 8. MAIN AREA
cur_subj = st.session_state.selected_subject
subj_meta = SUBJECT_METADATA.get(cur_subj, {"title": cur_subj if cur_subj != "All Subjects" else "All Subjects", "icon": "🌐", "type": "General"})
subj_icon = subj_meta.get("icon", "📚")
subj_title = subj_meta.get("title", cur_subj)

st.markdown(
    f"""<div class="subj-banner">
        {subj_icon} <span>{subj_title}</span>
        <span class="subj-tag">{cur_subj}</span>
    </div>""",
    unsafe_allow_html=True,
)

#  Render existing messages 
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "✨"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

#  Empty chat: show subject overview + sample questions 
if len(st.session_state.messages) == 0:
    st.markdown(f"### 💡 You're studying **{subj_title}**")

    if cur_subj != "All Subjects":
        st.markdown(
            f"You selected **{subj_icon} {subj_title}**. "
            f"All your questions will be answered strictly from **{subj_title}** notes and lab materials. "
            f"Switch to a different subject from the sidebar anytime — your chat here will be saved!"
        )
    else:
        st.markdown(
            "You're in **Global Search** mode — questions are answered across all 38 subjects. "
            "Select a specific subject from the sidebar for focused answers."
        )

    st.markdown("---")
    st.markdown("#### 💬 Sample questions you can ask:")

    sample_qs = SUBJECT_SAMPLE_QUESTIONS.get(cur_subj, SUBJECT_SAMPLE_QUESTIONS["All Subjects"])

    # Show 4 sample questions in a 2×2 grid
    col1, col2 = st.columns(2)
    cols = [col1, col2, col1, col2]
    for i, q in enumerate(sample_qs[:4]):
        if cols[i].button(f"💬 {q}", key=f"sample_q_{i}", use_container_width=True):
            st.session_state.prompt_input = q
            st.rerun()


# PROMPT TEMPLATE
# Minimum context length (chars) before we consider it "found"
_MIN_CONTEXT_LENGTH = 80

PROMPT_TEMPLATE = """You are a STRICT academic assistant for engineering students.

Current Subject: {active_subject}

=== ABSOLUTE RULES (NO EXCEPTIONS) ===
1. Answer ONLY using the context extracted from the {active_subject} notes below.
2. If the context does NOT contain information to answer the question, reply with ONLY this single sentence:
   "I don't have this topic in the provided notes for {active_subject}."
   Then STOP. Do NOT write anything else. Do NOT add general knowledge. Do NOT say 'however'. Do NOT provide any additional information.
3. NEVER use your training knowledge to fill gaps. NEVER answer from memory.
4. Do NOT answer questions that are about a different subject. If a student asks about Python while on AI subject, reply with only: "I don't have this topic in the provided notes for {active_subject}."
5. Only if the context is sufficient: answer thoroughly using Markdown headings (##), bullet points, bold key terms.

Chat History:
{chat_history}

Context from {active_subject} Notes:
{context}

Student Question:
{question}

Answer (from notes only, or the not-found sentence, nothing else):
"""

#  HANDLE INPUT & STREAM RESPONSE
input_text = st.session_state.pop("prompt_input", None)
user_query = st.chat_input(f"Ask anything about {subj_title}...") or input_text

if user_query:
    if len(st.session_state.messages) == 0:
        st.session_state.session_title = user_query[:28]

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)

    recent_turns = st.session_state.messages[-7:-1]
    history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in recent_turns])

    with st.chat_message("assistant", avatar="✨"):
        try:
            with st.spinner("Thinking..."):
                context_str = retrieve_context(user_query, subject_filter=cur_subj, k=6)

            # ── HARD GUARD: if context is empty or too short, skip LLM entirely ──
            if len(context_str.strip()) < _MIN_CONTEXT_LENGTH:
                not_found_msg = f"I don't have this topic in the provided notes for {cur_subj}."
                st.markdown(not_found_msg)
                st.session_state.messages.append({"role": "assistant", "content": not_found_msg})
                save_session(
                    st.session_state.active_session_id,
                    st.session_state.session_title,
                    st.session_state.messages,
                    subject=cur_subj,
                )
            else:
                prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                llm = ChatOllama(model=selected_model, temperature=0.0)
                chain = prompt | llm

                response_container = st.empty()
                full_response = ""

                for chunk in chain.stream({
                    "active_subject": cur_subj,
                    "context": context_str,
                    "chat_history": history_str,
                    "question": user_query,
                }):
                    piece = chunk.content if hasattr(chunk, "content") else str(chunk)
                    full_response += piece
                    response_container.markdown(full_response + "▌")

                response_container.markdown(full_response)

                st.session_state.messages.append({"role": "assistant", "content": full_response})
                save_session(
                    st.session_state.active_session_id,
                    st.session_state.session_title,
                    st.session_state.messages,
                    subject=cur_subj,
                )

        except Exception as e:
            st.error(f"❌ Error: {e}")
