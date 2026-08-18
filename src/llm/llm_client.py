import os
import socket
from typing import List, Generator, Dict, Any, Optional, Callable

try:
    import ollama
except ImportError:  # pragma: no cover - optional dependency for local model listing
    ollama = None

from langchain_core.prompts import ChatPromptTemplate

try:
    from langchain_ollama import ChatOllama
except ImportError:  # pragma: no cover - optional dependency for local inference
    ChatOllama = None

try:
    from langchain_groq import ChatGroq
except ImportError:  # pragma: no cover - optional dependency for cloud inference
    ChatGroq = None

from src.prompts.prompt_templates import PROMPT_TEMPLATE
from src.utils.helpers import load_app_config, get_groq_api_key


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


def get_local_ollama_models() -> List[str]:
    """Fetch locally installed Ollama models only if the Ollama daemon is active."""
    if not is_ollama_online() or ollama is None:
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
) -> Optional[ChatOllama]:
    """Create a LangChain ChatOllama instance if server is online."""
    if ChatOllama is None or not is_ollama_online():
        return None
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
    Unified streaming generator that manages Groq Cloud and Ollama Local inference with zero unhandled connection crashes.
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
    groq_error_detail = ""

    # 1. ATTEMPT GROQ CLOUD
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
            return
        except Exception as groq_err:
            groq_error_detail = str(groq_err)
            if "Auto" in selected_engine:
                if on_fallback:
                    on_fallback()
            else:
                yield f"⚠️ **Groq Cloud Error:** {groq_err}\n\nPlease check your `GROQ_API_KEY` or switch AI Engine."
                return

    # 2. ATTEMPT LOCAL OLLAMA (If Groq failed or Ollama was explicitly selected)
    ollama_available = is_ollama_online()

    if ollama_available:
        if ChatOllama is None:
            yield (
                "⚠️ **Ollama package not installed in this environment.**\n\n"
                "Install the required dependency and restart the app: `pip install ollama langchain-ollama`."
            )
            return
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

    # 3. NEITHER ENGINE IS AVAILABLE -> CLEAR USER GUIDANCE (No Raw Sockets Crashing!)
    if not groq_api_key:
        yield (
            "⚠️ **Groq API Key Required**\n\n"
            "To get instant AI answers, please configure your **Groq API Key**:\n"
            "1. On **Streamlit Cloud**: Add `GROQ_API_KEY = \"gsk_...\"` in **App Settings ➡️ Secrets**.\n"
            "2. Locally: Add `GROQ_API_KEY=gsk_...` inside your `.env` file, or start local Ollama with `ollama serve`."
        )
    elif "Auto" in selected_engine and groq_error_detail:
        yield (
            f"⚠️ **Groq API Rate Limit or Network Issue:**\n\n`{groq_error_detail}`\n\n"
            "**Note:** Local Ollama is not running on this host (normal for Streamlit Cloud). "
            "Please wait 1-2 minutes for Groq rate limits to reset."
        )
    else:
        yield (
            "⚠️ **Local Ollama Not Running**\n\n"
            "Could not connect to Ollama at `http://localhost:11434`.\n\n"
            "- If running locally: Start Ollama by typing `ollama serve` in a terminal.\n"
            "- If on **Streamlit Cloud**: Select **⚡ Auto** or **🚀 Groq Cloud** in the sidebar."
        )
