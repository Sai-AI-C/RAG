import os
import fitz

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama


# Load Environment Variables & Config

load_dotenv()

print("Starting PDF RAG using Ollama...")

ollama_api_key = os.getenv("OLLAMA_API_KEY")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

if ollama_api_key:
    print("Ollama API Key loaded successfully.")
else:
    print(f"Connecting to Ollama server at: {ollama_base_url}")


# Embedding Model & Vector Database
print("\nLoading Embedding Model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

DB_PATH = "pdf_db/chromadb"

if os.path.exists(DB_PATH):
    print("\nLoading Existing Vector Database from:", DB_PATH)
    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_model,
        collection_name="Document__C"
    )
else:
    print("\nVector database not found. Building database from all documents in ./PDF_Data...")
    from Load import load_all_documents

    documents = load_all_documents("./PDF_Data")
    print(f"Loaded {len(documents)} document pages across all subfolders.")

    print("\nSplitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} Chunks.")

    print("\nCreating New Vector Database...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_PATH,
        collection_name="Document__C"
    )

print("Vector Database Ready!")


# Retriever

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)


def format_docs(docs):
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page", "Unknown")
        formatted.append(f"[Page {page}]\n{doc.page_content}")
    return "\n\n".join(formatted)


# Prompt

template = """
You are an assistant for a B.Tech student's uploaded notes.

Answer ONLY using the supplied context.

Important rules:
1. Use the uploaded notes as the primary source.
2. Do not invent syllabus topics.
3. If an abbreviation has multiple meanings, determine its meaning
   from the supplied context.
4. If the context does not contain enough information, say so.
5. Do not use general knowledge to fill missing information.
6. When possible, mention the source document and page number.

Context:
{context}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)


# Ollama LLM Config

ollama_kwargs = {
    "model": os.getenv("OLLAMA_MODEL", "llama3.2"),
    "temperature": 0,
    "base_url": ollama_base_url,
}

# If API key is present, pass headers
if ollama_api_key:
    ollama_kwargs["client_kwargs"] = {
        "headers": {"Authorization": f"Bearer {ollama_api_key}"}
    }

# Option to force CPU if GPU CUDA error occurs
if os.getenv("OLLAMA_FORCE_CPU", "false").lower() == "true":
    ollama_kwargs["num_gpu"] = 0

llm = ChatOllama(**ollama_kwargs)


# Build RAG Chain

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# Chat Loop

if __name__ == "__main__":
    print("\n==============================")
    print(" PDF RAG using OLLAMA ")
    print("==============================")

    while True:

        question = input("\nAsk a Question (type 'exit' to quit): ")

        if question.lower() == "exit":
            break

        print("\nSearching...\n")

        answer = rag_chain.invoke(question)

        print("\nAnswer:\n")
        print(answer)
