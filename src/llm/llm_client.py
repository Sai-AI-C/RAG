import os
import socket
from typing import List, Generator, Dict, Any, Optional, Callable
import ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
try:
    from langchain_groq import ChatGroq
    from groq import Groq
except ImportError:
    ChatGroq = None
    Groq = None

from src.prompts.prompt_templates import PROMPT_TEMPLATE
from src.utils.helpers import load_app_config, get_groq_api_key

# Complete list of Groq models (ordered by reliability & intelligence)
ALL_GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-20b",
    "groq/compound-mini",
    "groq/compound",
    "allam-2-7b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]


def is_ollama_online(host: str = "127.0.0.1", port: int = 11434, timeout: float = 0.5) -> bool:
    """Quick socket check to verify if the local Ollama server is running."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def get_available_groq_models(api_key: Optional[str] = None) -> List[str]:
    """Fetch all available chat models from the user's Groq account."""
    key = api_key or get_groq_api_key()
    if not key or not Groq:
        return ALL_GROQ_MODELS

    try:
        client = Groq(api_key=key)
        all_models = client.models.list()
        chat_models = []
        for m in all_models.data:
            mid = m.id
            if "whisper" not in mid and "guard" not in mid and "orpheus" not in mid:
                chat_models.append(mid)
        if chat_models:
            return chat_models
    except Exception:
        pass

    return ALL_GROQ_MODELS


def get_local_ollama_models() -> List[str]:
    """Fetch locally installed Ollama models only if the Ollama daemon is active."""
    if not is_ollama_online():
        return []
    try:
        models_list = ollama.list().get("models", [])
        names = [m.get("name") or m.get("model") for m in models_list if m]
        if names:
            return names
    except Exception:
        pass
    return ["llama3.2:latest", "qwen2.5:7b", "llama3.1:8b", "mistral:7b"]


def create_groq_client(
    model_name: str = "openai/gpt-oss-120b",
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
) -> Optional[ChatOllama]:
    """Create a LangChain ChatOllama instance if server is online."""
    if not is_ollama_online():
        return None
    return ChatOllama(model=model_name, temperature=temperature)


def _extract_selected_groq_model(selected_engine: str) -> Optional[str]:
    """Extract model ID if explicitly chosen in the dropdown."""
    for m in ALL_GROQ_MODELS:
        if m in selected_engine or f"({m})" in selected_engine:
            return m
    return None


def stream_llm_response(
    active_subject: str,
    context: str,
    question: str,
    chat_history: str = "",
    selected_engine: str = "⚡ Auto Cascading Pool (Rotates across all models)",
    local_model: str = "llama3.2:latest",
    on_fallback: Optional[Callable[[str, str], None]] = None
) -> Generator[str, None, None]:
    """
    Unified streaming generator with multi-model cascading fallback.
    When a Groq model completes its limit (RateLimit, 429, TPM/RPM cap) or 404,
    it automatically shifts to the next model in the pool until success.
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
    last_groq_error = ""

    # 1. ATTEMPT GROQ CLOUD CASCADE POOL
    if ("Groq" in selected_engine or "Auto" in selected_engine or "Cascade" in selected_engine or "Pool" in selected_engine) and groq_api_key and ChatGroq:
        # Build ordered queue of models to try
        explicit_model = _extract_selected_groq_model(selected_engine)
        if explicit_model:
            # Put explicit model first, followed by the rest as fallback
            model_queue = [explicit_model] + [m for m in ALL_GROQ_MODELS if m != explicit_model]
        else:
            model_queue = list(ALL_GROQ_MODELS)

        for idx, g_model in enumerate(model_queue):
            try:
                llm_groq = ChatGroq(
                    model=g_model,
                    groq_api_key=groq_api_key,
                    temperature=temp,
                    max_retries=1
                )
                chain_groq = prompt | llm_groq

                # Stream response chunks
                chunk_received = False
                for chunk in chain_groq.stream(input_payload):
                    piece = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if piece:
                        chunk_received = True
                        yield piece

                if chunk_received:
                    groq_succeeded = True
                    return

            except Exception as groq_err:
                last_groq_error = str(groq_err)
                next_model = model_queue[idx + 1] if idx + 1 < len(model_queue) else "Ollama Local"

                # Notify UI about automatic shift to next model
                if on_fallback:
                    try:
                        on_fallback(g_model, next_model)
                    except Exception:
                        pass

                # Continue loop to try the next Groq model in the pool
                continue

    # 2. ATTEMPT LOCAL OLLAMA (If all Groq models hit limits or Ollama selected)
    ollama_available = is_ollama_online()

    if ollama_available:
        try:
            llm_ollama = ChatOllama(model=local_model, temperature=temp)
            chain_ollama = prompt | llm_ollama

            for chunk in chain_ollama.stream(input_payload):
                piece = chunk.content if hasattr(chunk, "content") else str(chunk)
                yield piece
            return
        except Exception as ollama_err:
            yield f"⚠️ **Local Ollama Error:** {ollama_err}\n\nPlease ensure model `{local_model}` is pulled (`ollama pull {local_model}`)."
            return

    # 3. NO ENGINE AVAILABLE -> CLEAR GUIDANCE
    if not groq_api_key:
        yield (
            "⚠️ **Groq API Key Required**\n\n"
            "To get instant AI answers across all 9 models:\n"
            "1. On **Streamlit Cloud / Spaces**: Add `GROQ_API_KEY = \"gsk_...\"` in Secrets / Environment Variables.\n"
            "2. Locally: Add `GROQ_API_KEY=gsk_...` in `.env` or start local Ollama with `ollama serve`."
        )
    elif last_groq_error:
        yield (
            f"⚠️ **All Groq Models Limit Reached**\n\n`{last_groq_error}`\n\n"
            "The system cycled through all available Groq models. Please wait 1 minute for your Groq rate limits to refresh."
        )
    else:
        yield (
            "⚠️ **Local Ollama Not Running**\n\n"
            "Could not connect to Ollama at `http://localhost:11434`.\n\n"
            "- If running locally: Start Ollama with `ollama serve`.\n"
            "- If in the cloud: Add `GROQ_API_KEY` to enable instant cloud inference."
        )
