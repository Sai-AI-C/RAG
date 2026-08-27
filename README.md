# OmniDoc-RAG

> **Grounded Academic Assistant for Engineering Students**  
> A FastAPI and JavaScript RAG application powered by ChromaDB, Groq, and optional local Ollama inference.

**Deployment:** Render web service using Uvicorn

🔴 **Live Demo**: [Click here to try the app](https://omnidoc-rag-360.streamlit.app/)


## 🎯 Problem Statement
- Engineering students face fragmented course materials across 38+ subjects
- Manual search through PDFs is time-consuming and error-prone
- Need for context-aware, grounded answers specific to each subject

## 💡 Solution
A RAG system that ingests course PDFs, retrieves relevant content, and provides grounded answers with fallback capabilities and subject-scoped validation.

---

## 🏛️ Project Architecture

```text
c:\OmniDoc-RAG\
├── README.md               # Project documentation and setup guide
├── requirements.txt        # Python dependency specifications
├── .env                    # Environment variables (API keys, model settings)
├── .gitignore              # Git ignore rules
├── config.yaml             # Core application & RAG pipeline configuration
├── api.py                  # FastAPI production server and SSE chat API
├── main.py                 # CLI & Application entry point
├── app.py                  # Legacy Streamlit interface for local use
├── frontend/index.html     # Production browser interface
├── src/                    # Modular RAG Source Package
│   ├── __init__.py
│   ├── ingestion/          # Document loading (PDF, DOCX, PPTX with OCR fallback)
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── chunking/           # Text splitting & subject-enriched chunking
│   │   ├── __init__.py
│   │   └── chunker.py
│   ├── embeddings/         # Embedding generation via SentenceTransformers
│   │   ├── __init__.py
│   │   └── embedder.py
│   ├── vectordb/           # ChromaDB PersistentClient & Collection management
│   │   ├── __init__.py
│   │   └── vector_store.py
│   ├── retrieval/          # Query expansion, keyword retrieval & relevance gate
│   │   ├── __init__.py
│   │   └── retriever.py
│   ├── prompts/            # Grounded exam-ready prompt templates
│   │   ├── __init__.py
│   │   └── prompt_templates.py
│   ├── llm/                # Unified Groq Cloud & Ollama fallback client
│   │   ├── __init__.py
│   │   └── llm_client.py
│   ├── api/                # Programmatic query pipeline & routes
│   │   ├── __init__.py
│   │   └── routes.py
│   └── utils/              # Configuration loader, abbreviation maps & session helpers
│       ├── __init__.py
│       └── helpers.py
├── tests/                  # Automated test suite
│   ├── __init__.py
│   └── test_app.py
└── logs/                   # System and audit logs
    └── app.log
```

---

## 🛡️ Key Features & Grounding Safeguards

1. **Anti-Hallucination Relevance Gate**: Queries asked under a specific subject are validated against course notes. Out-of-scope queries (e.g. asking *Data Analysis* under *Java Programming*) are intercepted programmatically with guidance to the correct subject.
2. **Context-Aware Abbreviation Resolution**: Short queries like `CN`, `DA`, `JF`, `DFF` are resolved strictly within the active subject's scope (e.g., `CN` in *Software Testing* → *Control Flow Graph*, while `CN` in *Computer Networks* → *Computer Networks*).
3. **Resilient Inference Engine**: Uses Groq cloud models with account-aware model discovery, unavailable-model detection, rate-limit handling, and optional local Ollama fallback.
4. **Subject-Scoped Navigation**: Over 38 engineering subjects categorized across *AI & Data Science*, *Networks & Security*, *Core CS & Systems*, and *Management & Electives*.

## 📊 System Performance

| Metric | Value |
|---|---|
| Supported Subjects | 38+ engineering courses |
| Indexed Chunks | 36K+ course-document chunks |
| Retrieval | Subject-aware ChromaDB keyword search |
| Response Behavior | Streaming responses via Server-Sent Events |
| Supported File Formats | PDF, DOCX, PPTX + OCR fallback |

## 📈 Dataset & Course Coverage

| Category | Details |
|---|---|
| Total Subjects | 38+ engineering courses |
| AI & Data Science | 7 subjects (ML, NLP, RL, etc.) |
| Networks & Security | 6 subjects (CN, CNS, etc.) |
| Core CS & Systems | 8 subjects (OS, DBMS, COA, etc.) |
| Management & Electives | 14 subjects |
| Document Types | PDF, DOCX, PPTX with OCR support |

## 🏗️ RAG Pipeline Components

| Component | Purpose | Technology |
|---|---|---|
| Ingestion | Multi-format document loading | PyPDF2, python-docx, pptx |
| Chunking | Semantic text splitting | RecursiveCharacterTextSplitter |
| Embeddings | Dense vector representations | SentenceTransformers (all-MiniLM-L6-v2) |
| Vector DB | Persistent retrieval | ChromaDB |
| Retrieval | Subject-aware keyword search with query expansion | ChromaDB |
| LLM | Cloud inference with fallback handling | Groq (primary) / Ollama (optional fallback) |
| Safety Gate | Anti-hallucination validation | Subject-scoped relevance checks |


---

## 🚀 Quick Start Guide

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/saicharan-r02/OmniDoc-RAG.git
cd OmniDoc-RAG
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Setup
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Ingest Course Notes
Place your subject folders inside `PDF_Data/` and run:
```bash
python main.py --mode ingest
```

To rebuild the vector index after replacing or re-OCRing documents, use the explicit rebuild mode. It deletes only matching ChromaDB records; files inside `PDF_Data/` are never deleted:
```bash
# Rebuild one subject
python main.py --mode ingest --subject MSF --rebuild

# Rebuild the complete PDF_Data index
python main.py --mode ingest --rebuild
```

### 4. Launch the Render-style API locally
Start the FastAPI server:
```bash
python api.py
```

Open `http://localhost:10000` in a browser. The API also provides `/api/health`, `/api/subjects`, and `/api/chat`.

The older Streamlit interface can still be started locally with `streamlit run app.py`, but it is not the production deployment described here.

### 5. Run Test Suite
```bash
python main.py --mode test
```

## 📚 Example Workflows

### Example 1: Subject-Scoped Query
**Input:** "What is a deadlock?" (under *Operating Systems* subject)  
**Output:** Retrieves OS-specific deadlock definitions from course notes, explains with examples

### Example 2: Abbreviation Resolution
**Input:** "Explain DFF" (under *DBMS* subject)  
**Output:** Resolves DFF to database-relevant context, not general meaning

### Example 3: Out-of-Scope Detection
**Input:** "How do I build a React app?" (under *Data Structures* subject)  
**Output:** Politely redirects to appropriate subject or declines

## ⚡ Performance Optimizations

- **Groq Cloud API**: Primary cloud inference provider
- **Model fallback**: Discovers models available to the configured Groq key and skips unavailable or rate-limited models
- **Ollama fallback**: Optional local inference when Ollama is running
- **ChromaDB**: Persistent indexed course-document storage
- **Query expansion**: Resolves subject-specific abbreviations and common academic phrases
- **SSE keepalive**: Keeps Render connections open during retrieval and generation

## 🌐 Deployment Status
The production service is deployed on Render with:

- **Build command:** `pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt`
- **Start command:** `python api.py`
- **Required secrets:** `GROQ_API_KEY` and `VECTOR_DB_URL`
- **Runtime settings:** one Uvicorn worker and CPU thread limits for low-memory instances

`VECTOR_DB_URL` must point to a downloadable ZIP containing the ChromaDB files. The database is ignored by Git because it is too large to store in the repository.

After deployment, verify `/api/health` returns `"status": "ok"`, `"groq_configured": true`, and a non-zero `"total_chunks"` value.

