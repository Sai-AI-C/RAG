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


# =====================================================
# Load Environment Variables & Config
# =====================================================

load_dotenv()

print("Starting PDF RAG using Ollama...")

ollama_api_key = os.getenv("OLLAMA_API_KEY")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

if ollama_api_key:
    print("Ollama API Key loaded successfully.")
else:
    print(f"Connecting to Ollama server at: {ollama_base_url}")


# =====================================================
# PDF Location
# =====================================================

PDF_PATH = "data/CD Notes All Units.pdf"
DB_PATH = "pdf_db"

# =====================================================
# Embedding Model
# =====================================================

print("\nLoading Embedding Model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =====================================================
# Create / Load Chroma Database
# =====================================================

if os.path.exists(DB_PATH):

    print("\nLoading Existing Vector Database...")

    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_model
    )

else:

    print(f"\nLoading PDF : {PDF_PATH}")
    doc = fitz.open(PDF_PATH)
    documents = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": PDF_PATH,
                    "page": page_num + 1
                }
            )
        )

    print(f"Loaded {len(documents)} Pages")

    print("\nSplitting PDF...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} Chunks")

    print("\nCreating New Vector Database...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )

print("Vector Database Ready")


# =====================================================
# Retriever
# =====================================================

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


# =====================================================
# Prompt
# =====================================================

template = """
You are an expert Computer Science Tutor.

Answer the question thoroughly using ONLY the facts and concepts described in the provided context.
You may synthesize information across pages if relevant.

If the answer is truly not present or cannot be inferred from the provided context at all, say:
"I don't know based on the provided notes."

Context:
{context}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)


# =====================================================
# Ollama LLM Config
# =====================================================

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


# =====================================================
# Build RAG Chain
# =====================================================

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# =====================================================
# Chat Loop
# =====================================================

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