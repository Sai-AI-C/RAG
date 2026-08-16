import os
from typing import List, Generator, Dict, Any, Optional, Callable
import ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

from src.prompts.prompt_templates import PROMPT_TEMPLATE
from src.utils.helpers import load_app_config, get_groq_api_key


def get_local_ollama_models() -> List[str]:
    """Fetch all locally installed Ollama models with fallback options."""
    try:
        models_list = ollama.list().get("models", [])
        names = [m.get("name") or m.get("model") for m in models_list if m]
        if names:
            return names
    except Exception:
        pass
    return ["llama3.2:latest", "qwen2.5:7b", "llama3.1:8b", "mistral:7b"]


def create_groq_client(
    model_name: str = "llama-3.1-8b-instant",
    api_key: Optional[str] = None,
    temperature: float = 0.0
) -> Optional[Any]:
    """Create a LangChain ChatGroq instance."""
    if not ChatGroq:
        return None
    key = api_key or get_groq_api_key()
    if not key:
        return None
    return ChatGroq(model=model_name, groq_api_key=key, temperature=temperature)


def create_ollama_client(
    model_name: str = "llama3.2:latest",
    temperature: float = 0.0
) -> ChatOllama:
    """Create a LangChain ChatOllama instance."""
    return ChatOllama(model=model_name, temperature=temperature)


def stream_llm_response(
    active_subject: str,
    context: str,
    question: str,
    chat_history: str = "",
    selected_engine: str = "⚡ Auto (Groq Fast ➡️ Ollama Backup)",
    local_model: str = "llama3.2:latest",
    on_fallback: Optional[Callable[[], None]] = None
) -> Generator[str, None, None]:
    """
    Unified streaming generator that automatically manages Groq cloud speed and Ollama local fallback.
    Yields text chunks as they arrive.
    """
    cfg = load_app_config()
    temp = cfg.get("llm", {}).get("temperature", 0.0)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    input_payload = {
        "active_subject": active_subject,
        "context": context,
        "chat_history": chat_history,
        "question": question,
    }

    groq_api_key = get_groq_api_key()
    groq_succeeded = False

    # 1. Attempt Groq Cloud if requested
    if ("Groq" in selected_engine or "Auto" in selected_engine) and groq_api_key and ChatGroq:
        try:
            g_model = "llama-3.1-8b-instant"
            if "70b" in selected_engine:
                g_model = "llama-3.3-70b-versatile"

            llm_groq = ChatGroq(model=g_model, groq_api_key=groq_api_key, temperature=temp)
            chain_groq = prompt | llm_groq

            for chunk in chain_groq.stream(input_payload):
                piece = chunk.content if hasattr(chunk, "content") else str(chunk)
                yield piece

            groq_succeeded = True
        except Exception as groq_err:
            if "Auto" in selected_engine:
                if on_fallback:
                    on_fallback()
            else:
                raise groq_err

    # 2. Fallback to local Ollama
    if not groq_succeeded:
        llm_ollama = ChatOllama(model=local_model, temperature=temp)
        chain_ollama = prompt | llm_ollama

        for chunk in chain_ollama.stream(input_payload):
            piece = chunk.content if hasattr(chunk, "content") else str(chunk)
            yield piece
