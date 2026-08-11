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

from langchain_groq import ChatGroq


# Load Environment Variables

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in .env file")

print("Groq API Key Loaded Successfully")


# Embedding Model

print("\nLoading Embedding Model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load Vector Database / Fallback to Load.py
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
You are an expert Computer Science Tutor and academic assistant.

Your job is to give the MOST COMPLETE, DETAILED, and THOROUGH answer possible using ONLY the facts, definitions, explanations, and examples provided in the context below.

Follow these rules strictly:
1. Cover EVERY aspect of the question — definitions, types, working, advantages, disadvantages, examples, and comparisons if available in the context.
2. Structure your answer with clear headings, numbered lists, and sub-points.
3. Do NOT summarize or shorten the answer. Give the FULL explanation from the notes.
4. If the context has examples, formulas, or diagrams described in text, include ALL of them in your answer.
5. If the answer spans multiple topics, cover ALL of them completely.
6. Only if the answer is truly not present in the context, say exactly: "I don't know based on the provided notes."

Context:
{context}

Question:
{question}

Provide a long, structured, detailed academic answer:
"""

prompt = ChatPromptTemplate.from_template(template)


# Groq LLM

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


# RAG Chain

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# Chat Loop

if __name__ == "__main__":
    print("\n==============================")
    print(" PDF RAG using GROQ ")
    print("==============================")

    while True:
        question = input("\nAsk a Question (type 'exit' to quit): ")

        if question.lower() == "exit":
            break

        print("\nSearching...\n")

        answer = rag_chain.invoke(question)

        print("Answer:\n")
        print(answer)
