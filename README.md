# OmniDoc-RAG 📚✨

> **Grounded Academic Assistant for Engineering Students**  
> Powered by Hybrid Dense Retrieval (ChromaDB + SentenceTransformers) and Dual-Inference Engine (Groq Fast Cloud + Ollama Local Fallback).

---

## 🏛️ Project Architecture

```text
c:\OmniDoc-RAG\
├── README.md               # Project documentation and setup guide
├── requirements.txt        # Python dependency specifications
├── .env                    # Environment variables (API keys, model settings)
├── .gitignore              # Git ignore rules
├── config.yaml             # Core application & RAG pipeline configuration
├── Load.py                 # Ingestion pipeline runner
├── main.py                 # CLI & Application entry point
├── app.py                  # Streamlit web application & UI
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
│   ├── retrieval/          # Query expansion, hybrid retrieval & anti-hallucination gate
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

### 4. Launch Application
Start the Streamlit interface:
```bash
python main.py --mode app
```
Or directly:
```bash
streamlit run app.py
```

### 5. Run Test Suite
```bash
python main.py --mode test
```

---

## 🛡️ Key Features & Grounding Safeguards

1. **Anti-Hallucination Relevance Gate**: Queries asked under a specific subject are validated against course notes. Out-of-scope queries (e.g. asking *Data Analysis* under *Java Programming*) are intercepted programmatically with guidance to the correct subject.
2. **Context-Aware Abbreviation Resolution**: Short queries like `CN`, `DA`, `JF`, `DFF` are resolved strictly within the active subject's scope (e.g., `CN` in *Software Testing* → *Control Flow Graph*, while `CN` in *Computer Networks* → *Computer Networks*).
3. **Hybrid Inference Engine**: Blazing fast responses using Groq Cloud (`llama-3.1-8b-instant`), with automatic fallback to local Ollama when rate limits are reached.
4. **Subject-Scoped Navigation**: Over 38 engineering subjects categorized across *AI & Data Science*, *Networks & Security*, *Core CS & Systems*, and *Management & Electives*.
