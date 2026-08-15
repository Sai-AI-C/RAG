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
try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None
from dotenv import load_dotenv

load_dotenv()

def get_groq_api_key() -> str:
    """Fetch Groq API key from Streamlit secrets or .env."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "").strip()

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
    div[data-testid="stSidebarContent"].stButton button {
        text-align: left;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 0.88rem;
        font-weight: 500;
        transition: background 0.2s;
    }
    div[data-testid="stSidebarContent"].stButton button:hover {
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
SUBJECT_METADATA={
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

# Global abbreviation map: subject-name → subject abbreviations within that subject's notes
# These are INTRA-SUBJECT abbreviations (terms used inside the notes of that subject)
SUBJECT_ABBREV = {
    "STM Notes": {
        "cfg": "Control Flow Graph", "cn": "Control Flow Graph (CFG)",
        "dfg": "Data Flow Graph", "dff": "Data Flow",
        "jf": "Junction / Decision Flow (Domain Testing)",
        "tf": "Transaction Flowgraph", "bb": "Basic Block",
        "nra": "Node Reduction Algorithm", "mcdc": "Modified Condition / Decision Coverage",
        "dd": "Definition-Definition anomaly", "du": "Definition-Use",
        "ku": "Kill-Use", "dk": "Definition-Kill",
        "lcsaj": "Linear Code Sequence And Jump",
    },
    "ML notes": {
        "da": "Data Analysis", "ml": "Machine Learning",
        "svm": "Support Vector Machine", "knn": "K-Nearest Neighbors",
        "dt": "Decision Tree", "rf": "Random Forest",
        "nn": "Neural Network", "bp": "Backpropagation",
        "lr": "Linear Regression", "log": "Logistic Regression",
        "em": "Expectation Maximization", "pca": "Principal Component Analysis",
        "nb": "Naive Bayes", "adaboost": "Adaptive Boosting",
    },
    "ML Lab": {
        "svm": "Support Vector Machine", "knn": "K-Nearest Neighbors",
        "lr": "Linear Regression", "nb": "Naive Bayes",
    },
    "AI": {
        "bfs": "Breadth First Search", "dfs": "Depth First Search",
        "ids": "Iterative Deepening Search", "a*": "A Star Search",
        "csp": "Constraint Satisfaction Problem", "kb": "Knowledge Base",
        "pl": "Propositional Logic", "fol": "First Order Logic",
    },
    "Neural network and deep learning": {
        "ann": "Artificial Neural Network", "rnn": "Recurrent Neural Network",
        "cnn": "Convolutional Neural Network", "dnn": "Deep Neural Network",
        "lstm": "Long Short-Term Memory", "gru": "Gated Recurrent Unit",
        "gan": "Generative Adversarial Network", "ae": "Autoencoder",
        "nnd": "Neural Networks and Deep Learning", "nndl": "Neural Networks and Deep Learning",
        "bp": "Backpropagation", "rbf": "Radial Basis Function",
    },
    "CN Notes": {
        "cn": "Computer Networks", "osi": "Open Systems Interconnection model",
        "tcp": "Transmission Control Protocol", "udp": "User Datagram Protocol",
        "ip": "Internet Protocol", "http": "HyperText Transfer Protocol",
        "dns": "Domain Name System", "nat": "Network Address Translation",
        "mac": "Media Access Control", "arp": "Address Resolution Protocol",
    },
    "CNS": {
        "rsa": "RSA Public Key Cryptography", "aes": "Advanced Encryption Standard",
        "des": "Data Encryption Standard", "sha": "Secure Hash Algorithm",
        "pki": "Public Key Infrastructure", "ca": "Certificate Authority",
        "vpn": "Virtual Private Network",
    },
    "DBMS": {
        "er": "Entity Relationship", "1nf": "First Normal Form",
        "2nf": "Second Normal Form", "3nf": "Third Normal Form",
        "bcnf": "Boyce-Codd Normal Form", "sql": "Structured Query Language",
        "dml": "Data Manipulation Language", "ddl": "Data Definition Language",
        "acid": "Atomicity Consistency Isolation Durability",
    },
    "OS": {
        "fcfs": "First Come First Served scheduling", "sjf": "Shortest Job First",
        "rr": "Round Robin scheduling", "tlb": "Translation Lookaside Buffer",
        "mmu": "Memory Management Unit", "pcb": "Process Control Block",
        "ipc": "Inter-Process Communication", "vm": "Virtual Memory",
    },
    "DAA Notes": {
        "da": "Design and Analysis of Algorithms", "daa": "Design and Analysis of Algorithms",
        "dp": "Dynamic Programming", "bfs": "Breadth First Search",
        "dfs": "Depth First Search", "mst": "Minimum Spanning Tree",
        "tsp": "Travelling Salesman Problem",
    },
    "Software Engineering": {
        "sdlc": "Software Development Life Cycle", "srs": "Software Requirements Specification",
        "uml": "Unified Modelling Language", "dfd": "Data Flow Diagram",
        "er": "Entity Relationship Diagram",
    },
}

# Cross-subject abbreviation map: subject key (like 'CN Notes') → subject title
SUBJECT_NAME_ABBREV = {
    "cn": "Computer Networks", "cns": "Cryptography and Network Security",
    "cd": "Compiler Design", "ai": "Artificial Intelligence",
    "daa": "Design and Analysis of Algorithms", "dbms": "Database Management Systems",
    "os": "Operating Systems", "ml": "Machine Learning",
    "nnd": "Neural Networks and Deep Learning", "nndl": "Neural Networks and Deep Learning",
    "dl": "Deep Learning", "nlp": "Natural Language Processing",
    "flat": "Formal Languages and Automata Theory", "devops": "DevOps",
    "se": "Software Engineering", "stm": "Software Testing Methodologies",
    "dppm": "Data Preparation and Pattern Mining",
    "acs": "Advanced Communication Systems",
    "coa": "Computer Organization and Architecture",
    "sna": "Social Network Analysis",
}


def is_short_query(query: str) -> bool:
    """Returns True for single words or short abbreviation-style queries."""
    words = query.strip().split()
    return len(words) <= 3 or (len(words) <= 5 and all(len(w) <= 5 for w in words))


def expand_query(query: str, active_subject: str = "All Subjects") -> str:
    cleaned = query.strip().lower()
    # Fix common typos
    cleaned = cleaned.replace("regreesion", "regression").replace("algorithem", "algorithm")

    # 1. Try subject-specific abbreviation table first (most accurate)
    subj_abbrevs = SUBJECT_ABBREV.get(active_subject, {})
    for word in cleaned.split():
        if word in subj_abbrevs:
            return f"{query} ({subj_abbrevs[word]})"

    # 2. Specific subject overrides for very common single-word queries
    subject_overrides = {
        ("CN Notes", frozenset(["cn", "define cn", "what is cn"])):
            "Computer Networks architecture OSI TCP/IP model layers protocols",
        ("CN Lab", frozenset(["cn", "define cn"])):
            "Computer Networks lab experiments socket programming",
        ("AI", frozenset(["ai", "define ai", "what is ai"])):
            "Artificial Intelligence definitions agents turing test search",
        ("AI-NLP Lab", frozenset(["ai", "nlp"])):
            "Artificial Intelligence NLP lab experiments programs",
        ("ML notes", frozenset(["ml", "define ml", "what is ml", "types ml"])):
            "Machine Learning types supervised unsupervised classification regression",
        ("DAA Notes", frozenset(["da", "daa", "define da", "what is da"])):
            "Design and Analysis of Algorithms asymptotic notations time complexity",
        ("ACS Lab", frozenset(["acs", "define acs"])):
            "Advanced Communication Systems lab experiments record",
        ("Neural network and deep learning", frozenset(["nnd", "nndl", "types of nnd"])):
            "Neural Networks Deep Learning types feed-forward recurrent RNN CNN autoencoders",
        ("STM Notes", frozenset(["stm", "what is stm"])):
            "Software Testing Methodologies testing types coverage white box black box",
    }
    for (subj, terms), expansion in subject_overrides.items():
        if active_subject == subj and cleaned in terms:
            return expansion

    # 3. Word-level expansion using subject name abbreviations (lowest priority)
    expanded_parts = []
    for word in cleaned.split():
        if word in SUBJECT_NAME_ABBREV:
            expanded_parts.append(SUBJECT_NAME_ABBREV[word])
    if expanded_parts:
        return f"{query} {' '.join(expanded_parts)}"

    return query

# 3. PERSISTENT SESSION STORAGE (PER-SUBJECT)
def _session_path(session_id: str) -> str:
    return os.path.join(CHAT_SESSIONS_DIR, f"{session_id}.json")

def get_all_sessions():
    sessions=[]
    for fname in os.listdir(CHAT_SESSIONS_DIR):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(CHAT_SESSIONS_DIR, fname), "r", encoding="utf-8") as f:
                    sessions.append(json.load(f))
            except Exception:
                pass
    sessions.sort(key=lambda x:x.get("timestamp",0),reverse=True)
    return sessions

def load_session(session_id: str):
    p=_session_path(session_id)
    if os.path.exists(p):
        try:
            with open(p,"r",encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None

def save_session(session_id: str,title: str,messages: list,subject: str="All Subjects"):
    if not messages:
        return
    data={
        "session_id":session_id,
        "title":title[:30]+("..." if len(title)>30 else ""),
        "timestamp":time.time(),
        "messages":messages,
        "subject":subject,
    }
    with open(_session_path(session_id),"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)


def delete_session(session_id: str):
    p=_session_path(session_id)
    if os.path.exists(p):
        os.remove(p)

# 4. VECTOR STORE & MODELS
def download_and_extract_db(download_url: str, target_dir: str) -> bool:
    """Download vector database zip archive and extract it."""
    import zipfile
    import urllib.request
    import re

    zip_path = "temp_pdf_db.zip"
    try:
        # Handle Google Drive share links
        if "drive.google.com" in download_url:
            file_id_match = re.search(r"/d/([a-zA-Z0-9_-]+)", download_url) or re.search(r"id=([a-zA-Z0-9_-]+)", download_url)
            if file_id_match:
                file_id = file_id_match.group(1)
                try:
                    import gdown
                    gdown.download(id=file_id, output=zip_path, quiet=False)
                except Exception:
                    direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                    urllib.request.urlretrieve(direct_url, zip_path)
            else:
                urllib.request.urlretrieve(download_url, zip_path)
        else:
            urllib.request.urlretrieve(download_url, zip_path)

        if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1000:
            return False

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(".")

        if os.path.exists(zip_path):
            os.remove(zip_path)
        return True
    except Exception as e:
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except Exception:
                pass
        st.error(f"❌ Error downloading database: {e}")
        return False


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def get_vector_store():
    persist_dir = "./pdf_db/chromadb"
    db_file = os.path.join(persist_dir, "chroma.sqlite3")

    if not os.path.exists(db_file):
        # Check if VECTOR_DB_URL is available in Streamlit Secrets or Environment
        download_url = None
        try:
            if "VECTOR_DB_URL" in st.secrets:
                download_url = st.secrets["VECTOR_DB_URL"]
        except Exception:
            pass
        if not download_url:
            download_url = os.getenv("VECTOR_DB_URL", "").strip()

        if download_url:
            with st.spinner("📦 Downloading knowledge base for cloud setup (~15-30s)... Please wait."):
                success = download_and_extract_db(download_url, persist_dir)
                if not success or not os.path.exists(db_file):
                    st.error("Vector database extraction failed. Please check the VECTOR_DB_URL in Streamlit secrets.")
                    st.stop()
        else:
            st.error(
                "⚠️ **Vector Database not found at `./pdf_db/chromadb`!**\n\n"
                "**For Streamlit Cloud Deployment:**\n"
                "1. Run `python create_db_zip.py` on your laptop to create `pdf_db.zip`.\n"
                "2. Upload `pdf_db.zip` to Google Drive or GitHub Releases and get the share link.\n"
                "3. In Streamlit Cloud dashboard ➡️ **Settings (⚙️) ➡️ Secrets**, add:\n"
                "```toml\nVECTOR_DB_URL = \"https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing\"\n```\n"
                "4. Reboot the app!"
            )
            st.stop()

    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(name="Document__C")


@st.cache_data(show_spinner=False)
def get_subject_counts():
    collection=get_vector_store()
    try:
        results=collection.get(include=["metadatas"])
        counts={}
        for m in results.get("metadatas",[]):
            if m and "subject" in m:
                counts[m["subject"]]=counts.get(m["subject"],0)+1
        return counts
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def get_local_ollama_models():
    try:
        models_list=ollama.list().get("models",[])
        names=[m.get("name") or m.get("model") for m in models_list if m]
        if names:
            return names
    except Exception:
        pass
    return ["qwen2.5:7b","llama3.2:latest","llama3.1:8b","mistral:7b"]

# 5. RAG RETRIEVAL WITH SUBJECT FILTERING
def get_related_subjects(subject: str) -> list:
    if not subject or subject == "All Subjects":
        return []
    related = [subject]
    if "Notes" in subject:
        related.append(subject.replace("Notes", "Lab").strip())
    elif "Lab" in subject:
        related.append(subject.replace("Lab", "Notes").strip())
    
    # Cross-disciplinary subject linkages in curriculum
    extra = {
        "AI": ["AI-NLP Lab", "NLP", "ML notes"],
        "AI-NLP Lab": ["AI", "NLP"],
        "ML notes": ["ML Lab", "Neural network and deep learning", "AI"],
        "ML Lab": ["ML notes", "Neural network and deep learning"],
        "Neural network and deep learning": ["ML notes", "ML Lab", "Reinforcement Learning"],
        "Reinforcement Learning": ["Neural network and deep learning", "ML notes"],
        "NLP": ["AI", "AI-NLP Lab"],
        "DBMS": ["DBMS Lab"],
        "DBMS Lab": ["DBMS"],
        "Devops": ["Devops lab"],
        "Devops lab": ["Devops"],
        "Java": ["Java Lab"],
        "Java Lab": ["Java"],
        "Data structure": ["DAA Notes"],
        "DAA Notes": ["Data structure"],
        "CD Notes": ["FLAT"],
        "FLAT": ["CD Notes"],
        "CN Notes": ["CN Lab"],
        "CN Lab": ["CN Notes"],
        "CNS": ["CNS Lab"],
        "CNS Lab": ["CNS"],
    }
    if subject in extra:
        val = extra[subject]
        if isinstance(val, list):
            related.extend(val)
        else:
            related.append(val)
    return list(set(related))


def retrieve_context(query: str, subject_filter: str = "All Subjects", k: int = 8) -> str:
    """Hybrid retrieval: semantic search + keyword search for short/abbreviation queries."""
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

    # --- Semantic search (always done) ---
    sem_results = collection.query(query_embeddings=[query_embedding], n_results=k, where=where_clause)
    sem_docs = sem_results.get("documents", [[]])[0]

    # --- Keyword search for short / abbreviation queries ---
    # For queries like "CN", "DA", "JF", embedding similarity is weak.
    # ChromaDB where_document $contains does substring matching on chunk text.
    kw_docs = []
    raw = query.strip()
    words = raw.split()
    if is_short_query(raw) and len(raw) >= 2:
        try:
            # Search for the exact query term in document text
            kw_filter = {"$contains": raw.upper()} if len(raw) <= 4 else {"$contains": raw}
            kw_where = {"$and": [{"subject": where_clause["subject"]}, {"$document": kw_filter}]} \
                if where_clause and "subject" in where_clause else {"$document": kw_filter}

            # ChromaDB keyword search via where_document
            kw_res = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, 5),
                where=where_clause,
                where_document={"$contains": raw.upper() if len(raw) <= 4 else raw},
            )
            kw_docs = kw_res.get("documents", [[]])[0]
        except Exception:
            kw_docs = []

    # --- Merge: keyword hits first (more precise), then semantic hits, deduplicate ---
    seen = set()
    merged = []
    for doc in (kw_docs + sem_docs):
        key = doc[:120]  # deduplicate by first 120 chars
        if key not in seen:
            seen.add(key)
            merged.append(doc)

    return "\n\n---\n\n".join(merged)

# 6. SESSION STATE INITIALIZATION
def _fresh_session_state(subject="All Subjects"):
    return {
        "active_session_id": uuid.uuid4().hex[:8],
        "messages":[],
        "session_title":"New Chat",
        "selected_subject":subject,
    }

if "active_session_id" not in st.session_state:
    st.session_state.update(_fresh_session_state())
    st.session_state.loaded_session_id=None

# Load session from disk only on session ID change
if st.session_state.get("loaded_session_id")!=st.session_state.active_session_id:
    data = load_session(st.session_state.active_session_id)
    if data:
        st.session_state.messages=data.get("messages",[])
        st.session_state.session_title=data.get("title","Chat Session")
        st.session_state.selected_subject=data.get("subject","All Subjects")
    else:
        st.session_state.messages=[]
        st.session_state.session_title="New Chat"
        # Keep current selected_subject so switching subject → new chat keeps subject
    st.session_state.loaded_session_id=st.session_state.active_session_id

def switch_subject(new_subject: str):
    """Save current chat, then open/create a chat session for new_subject."""
    cur_subject=st.session_state.selected_subject

    # Save current chat if it has messages and subject is changing
    if st.session_state.messages and cur_subject!=new_subject:
        save_session(
            st.session_state.active_session_id,
            st.session_state.session_title,
            st.session_state.messages,
            subject=cur_subject,
        )

    # Look for an existing unfinished session for this subject
    all_sessions=get_all_sessions()
    matching=[s for s in all_sessions if s.get("subject")==new_subject]

    if matching and len(matching[0].get("messages",[]))==0:
        # Restore empty session
        s = matching[0]
        st.session_state.active_session_id=s["session_id"]
        st.session_state.messages=[]
        st.session_state.session_title="New Chat"
    else:
        # Start a fresh session for this subject
        new_id = uuid.uuid4().hex[:8]
        st.session_state.active_session_id=new_id
        st.session_state.messages=[]
        st.session_state.session_title="New Chat"

    st.session_state.selected_subject=new_subject
    st.session_state.loaded_session_id=st.session_state.active_session_id

# 7. SIDEBAR — SUBJECT LIST & CHAT HISTORY
SIDEBAR_CATEGORIES={
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
        cur_subj=st.session_state.selected_subject
        if st.session_state.messages:
            save_session(
                st.session_state.active_session_id,
                st.session_state.session_title,
                st.session_state.messages,
                subject=cur_subj,
            )
        new_id=uuid.uuid4().hex[:8]
        st.session_state.active_session_id=new_id
        st.session_state.loaded_session_id=new_id
        st.session_state.messages=[]
        st.session_state.session_title="New Chat"
        st.rerun()
    st.divider()

    # SUBJECT SELECTOR 
    st.markdown("**📚 Select Subject**")
    cur_subj=st.session_state.selected_subject

    # All Subjects button
    btn_type="primary" if cur_subj=="All Subjects" else "secondary"
    if st.button("🌐 All Subjects",use_container_width=True,type=btn_type,key="btn_all"):
        if cur_subj!="All Subjects":
            switch_subject("All Subjects")
            st.rerun()

    # Per-category subject buttons
    for cat_name,subjects in SIDEBAR_CATEGORIES.items():
        st.markdown(f"<div class='sidebar-cat'>{cat_name}</div>", unsafe_allow_html=True)
        for subj in subjects:
            meta=SUBJECT_METADATA.get(subj, {})
            icon=meta.get("icon", "📖")
            title=meta.get("title", subj)
            stype=meta.get("type", "")
            label=f"{icon} {title}"
            if stype=="Lab":
                label+=" 🔬"

            is_active=(cur_subj==subj)
            btn_style="primary" if is_active else "secondary"
            if st.button(label,key=f"sbtn_{subj}",use_container_width=True,type=btn_style):
                if not is_active:
                    switch_subject(subj)
                    st.rerun()
    st.divider()

#CHAT HISTORY 
    st.markdown("**💬 Chat History**")
    saved=get_all_sessions()
    if saved:
        for s in saved:
            s_id=s["session_id"]
            s_title=s.get("title", "Chat")
            s_subj=s.get("subject", "All Subjects")
            is_active=(s_id==st.session_state.active_session_id)
            col_t,col_d=st.columns([0.82,0.18])
            icon="📌" if is_active else "💬"
            subj_meta=SUBJECT_METADATA.get(s_subj, {})
            subj_icon=subj_meta.get("icon", "📚")
            label=f"{icon} {s_title}\n{subj_icon} {s_subj}"
            if col_t.button(label,key=f"sess_{s_id}",use_container_width=True):
                if s_id!=st.session_state.active_session_id:
                    st.session_state.active_session_id = s_id
                    st.session_state.messages=s.get("messages",[])
                    st.session_state.session_title=s_title
                    st.session_state.selected_subject=s_subj
                    st.session_state.loaded_session_id=s_id
                    st.rerun()
            if col_d.button("🗑️",key=f"del_{s_id}"):
                delete_session(s_id)
                if s_id==st.session_state.active_session_id:
                    new_id=uuid.uuid4().hex[:8]
                    st.session_state.active_session_id=new_id
                    st.session_state.loaded_session_id=new_id
                    st.session_state.messages=[]
                    st.session_state.session_title="New Chat"
                st.rerun()
    else:
        st.caption("No saved chats yet.")

    st.divider()

    # AI ENGINE / MODEL SELECTOR 
    st.markdown("**🧠 AI Engine**")
    groq_api_key=get_groq_api_key()
    local_models=get_local_ollama_models()

    engine_options=[]
    if groq_api_key:
        engine_options.append("⚡ Auto (Groq Fast ➡️ Ollama Backup)")
        engine_options.append("🚀 Groq Cloud (llama-3.1-8b-instant)")
        engine_options.append("🚀 Groq Cloud (llama-3.3-70b-versatile)")
    engine_options.append("💻 Ollama Local (Unlimited)")

    selected_engine=st.selectbox("AI Engine",engine_options,index=0,label_visibility="collapsed")
    
    selected_ollama_model="llama3.2:latest"
    if "Ollama" in selected_engine or "Auto" in selected_engine:
        if local_models:
            selected_ollama_model=st.selectbox("Local Model", local_models, index=0)

# 8. MAIN AREA
cur_subj=st.session_state.selected_subject
subj_meta=SUBJECT_METADATA.get(cur_subj, {"title": cur_subj if cur_subj != "All Subjects" else "All Subjects", "icon": "🌐", "type": "General"})
subj_icon=subj_meta.get("icon", "📚")
subj_title=subj_meta.get("title", cur_subj)

st.markdown(
    f"""<div class="subj-banner">
        {subj_icon} <span>{subj_title}</span>
        <span class="subj-tag">{cur_subj}</span>
    </div>""",
    unsafe_allow_html=True,
)

#  Render existing messages 
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"]=="user" else "✨"
    with st.chat_message(msg["role"],avatar=avatar):
        st.markdown(msg["content"])

#  Empty chat: show subject overview + sample questions 
if len(st.session_state.messages)==0:
    st.markdown(f"### 💡 You're studying **{subj_title}**")

    if cur_subj!="All Subjects":
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

    sample_qs=SUBJECT_SAMPLE_QUESTIONS.get(cur_subj, SUBJECT_SAMPLE_QUESTIONS["All Subjects"])

    # Show 4 sample questions in a 2×2 grid
    col1,col2=st.columns(2)
    cols=[col1,col2,col1,col2]
    for i, q in enumerate(sample_qs[:4]):
        if cols[i].button(f"💬 {q}",key=f"sample_q_{i}",use_container_width=True):
            st.session_state.prompt_input=q
            st.rerun()

# PROMPT TEMPLATE
# Minimum context length (chars) before we consider it "found"
_MIN_CONTEXT_LENGTH = 80

PROMPT_TEMPLATE = """You are OmniDoc AI, an academic assistant for engineering students.

Your answers must be grounded primarily in the retrieved course notes for the ACTIVE SUBJECT.

ACTIVE SUBJECT: {active_subject}

CORE RULE:
The retrieved course notes are the source of truth for subject-specific definitions,
terminology, abbreviations, algorithms, procedures, formulas, and concepts.

Do NOT answer from memory merely because you know a possible meaning of a term.
Do NOT invent information not supported by the retrieved course notes.
Do NOT substitute a common textbook meaning for the meaning used in the student's course.

ABBREVIATION / SHORT QUERY RULE:
When the student asks a short question like "CN", "JF", "DA", "DFF":
1. Search the retrieved context for the exact abbreviation text.
2. Look for an explicit expansion such as: "CN stands for ...", "CN = ...", or "Control Network (CN)".
3. Determine if the retrieved context associates the abbreviation with a specific concept.
4. Only use that meaning if supported by the context.
If the abbreviation cannot be established from the retrieved notes, say:
  "The provided {active_subject} notes do not clearly define '[ABBREV]' as an abbreviation."
Do NOT guess from general knowledge or other subjects.

GROUNDING:
- SUPPORTED: Directly stated in the course notes → state as fact.
- INFERRED: Reasonably derived from the notes → prefix with "Based on the provided notes, ..."
- UNKNOWN: Not in the notes → say "The retrieved notes do not establish this."
Never present UNKNOWN information as fact.

CONCEPT SEPARATION:
Do NOT merge different concepts because they appear in the same retrieved context.
Explain Control Flow Graph, Data Flow Graph, Domain Testing etc. as separate topics unless
the notes explicitly relate them.

ALGORITHM SAFETY:
When explaining an algorithm or procedure:
- Use only steps supported by the notes, in the order given.
- Do NOT invent missing steps or "complete" an incomplete algorithm from assumptions.
- If the notes contain only part of the procedure, state that explicitly.

EXAM ANSWER FORMAT:
For "explain X":
  ## X
  ### Definition
  ### Explanation
  ### Key Points
  ### Example  ← only if the notes contain one, or label it "Illustrative example"
  ### Exam Point
For multiple concepts ("explain A, B and C"): answer each under its own ## heading.
For comparisons: use a table.
Do NOT create sections that are not relevant.

EXAM ACCURACY:
A shorter correct answer is better than a longer unsupported one.
Do NOT make the answer sound authoritative when evidence is weak.

OUT-OF-SCOPE:
If the requested concept is not supported by the retrieved notes for {active_subject},
do not fabricate an answer. Say the context does not establish it.

CHAT HISTORY RULE:
Chat history is only for resolving references like "explain the previous one", "compare them".
Chat history must NOT be treated as course-note evidence.

BEFORE ANSWERING, verify:
1. Did I answer the exact question?
2. Is the answer supported by the retrieved notes?
3. Did I guess an abbreviation meaning?
4. Did I accidentally use knowledge from a different subject?
5. Did I mix two different concepts?
6. Did I invent an algorithm step or example?
7. Did I clearly state when information is missing?

Do NOT start with "Certainly!", "Sure!", or "Of course!".

RETRIEVED COURSE NOTES ({active_subject}):
{context}

CHAT HISTORY:
{chat_history}

STUDENT QUESTION:
{question}

GROUNDED ANSWER:
"""

#  HANDLE INPUT & STREAM RESPONSE
input_text=st.session_state.pop("prompt_input",None)
user_query=st.chat_input(f"Ask anything about {subj_title}...") or input_text

if user_query:
    if len(st.session_state.messages)==0:
        st.session_state.session_title=user_query[:28]

    st.session_state.messages.append({"role":"user","content":user_query})
    with st.chat_message("user",avatar="👤"):
        st.markdown(user_query)

    recent_turns=st.session_state.messages[-7:-1]
    history_str="\n".join([f"{m['role'].capitalize()}:{m['content']}" for m in recent_turns])

    with st.chat_message("assistant",avatar="✨"):
        try:
            with st.spinner("Thinking..."):
                context_str=retrieve_context(user_query, subject_filter=cur_subj, k=8)

            # HARD GUARD: if context is empty or too short,skip LLM entirely 
            if len(context_str.strip())<_MIN_CONTEXT_LENGTH:
                not_found_msg=f"I don't have this topic in the provided notes for {cur_subj}."
                st.markdown(not_found_msg)
                st.session_state.messages.append({"role":"assistant","content":not_found_msg})
                save_session(
                    st.session_state.active_session_id,
                    st.session_state.session_title,
                    st.session_state.messages,
                    subject=cur_subj,
                )
            else:
                prompt=ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                input_payload={
                    "active_subject":cur_subj,
                    "context":context_str,
                    "chat_history":history_str,
                    "question":user_query,
                }

                response_container=st.empty()
                full_response=""
                groq_succeeded=False

                # ── ATTEMPT 1: Groq Cloud (Fast 500 tokens/sec) ──────
                if ("Groq" in selected_engine or "Auto" in selected_engine) and groq_api_key and ChatGroq:
                    try:
                        g_model = "llama-3.1-8b-instant"
                        if "70b" in selected_engine:
                            g_model = "llama-3.3-70b-versatile"

                        llm_groq = ChatGroq(model=g_model, groq_api_key=groq_api_key, temperature=0.2)
                        chain_groq = prompt | llm_groq

                        for chunk in chain_groq.stream(input_payload):
                            piece = chunk.content if hasattr(chunk, "content") else str(chunk)
                            full_response += piece
                            response_container.markdown(full_response + "▌")

                        response_container.markdown(full_response)
                        groq_succeeded = True
                    except Exception as groq_err:
                        if "Auto" in selected_engine:
                            st.caption("⚡ *Groq rate limit reached — automatically switched to local Ollama.*")
                            full_response = ""
                        else:
                            raise groq_err

                # ── ATTEMPT 2: Local Ollama Fallback (100% Unlimited) ─
                if not groq_succeeded:
                    llm_ollama = ChatOllama(model=selected_ollama_model, temperature=0.2)
                    chain_ollama = prompt | llm_ollama

                    for chunk in chain_ollama.stream(input_payload):
                        piece = chunk.content if hasattr(chunk, "content") else str(chunk)
                        full_response += piece
                        response_container.markdown(full_response + "▌")

                    response_container.markdown(full_response)

                st.session_state.messages.append({"role":"assistant","content":full_response})
                save_session(
                    st.session_state.active_session_id,
                    st.session_state.session_title,
                    st.session_state.messages,
                    subject=cur_subj,
                )

        except Exception as e:
            st.error(f"❌ Error:{e}")