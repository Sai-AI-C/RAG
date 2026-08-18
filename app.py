import os
import uuid
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import clean architecture modules from src
from src.utils.helpers import (
    load_app_config,
    get_groq_api_key,
    SUBJECT_METADATA,
    SIDEBAR_CATEGORIES,
    SUBJECT_SAMPLE_QUESTIONS,
    get_all_sessions,
    save_session,
    delete_session,
)
from src.vectordb.vector_store import get_vector_store, get_subject_counts
from src.retrieval.retriever import retrieve_context, is_context_relevant
from src.llm.llm_client import get_local_ollama_models, is_ollama_online, stream_llm_response

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
        transition: all 0.2s ease;
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
        padding: 14px 20px;
        border-radius: 12px;
        margin-bottom: 18px;
        font-weight: 600;
        font-size: 1.1rem;
        display: flex;
        align-items: center;
        gap: 12px;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2);
    }
    .subj-banner .subj-tag {
        font-size: 0.8rem;
        background: rgba(255, 255, 255, 0.22);
        padding: 3px 12px;
        border-radius: 20px;
        font-weight: 500;
        margin-left: auto;
    }
    /* Subject category label in sidebar */
    .sidebar-cat {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        padding: 8px 0 3px 4px;
    }
    /* Chat bubbles */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 8px;
    }
    /* Mobile responsive */
    @media (max-width: 768px) {
        .subj-banner { flex-direction: column; align-items: flex-start; gap: 6px; }
        .subj-banner .subj-tag { margin-left: 0; }
        .stButton button { min-height: 40px; }
    }
</style>
""", unsafe_allow_html=True)


# 2. SESSION STATE INITIALIZATION
def init_session_state():
    if "active_session_id" not in st.session_state:
        st.session_state.active_session_id = uuid.uuid4().hex[:8]
    if "selected_subject" not in st.session_state:
        st.session_state.selected_subject = "All Subjects"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_title" not in st.session_state:
        st.session_state.session_title = "New Chat"
    if "prompt_input" not in st.session_state:
        st.session_state.prompt_input = None

init_session_state()


def switch_subject(new_subject: str):
    """Switch active subject and start a fresh chat session."""
    st.session_state.selected_subject = new_subject
    st.session_state.active_session_id = uuid.uuid4().hex[:8]
    st.session_state.messages = []
    st.session_state.session_title = "New Chat"
    st.session_state.prompt_input = None


# 3. SIDEBAR NAVIGATION
with st.sidebar:
    st.markdown("### 📚 **OmniDoc AI**")
    st.caption("Grounded Academic Assistant for Engineering Students")

    col_new, col_cnt = st.columns([0.65, 0.35])
    if col_new.button("➕ New Chat", use_container_width=True, type="primary"):
        switch_subject(st.session_state.selected_subject)
        st.rerun()

    subj_counts = get_subject_counts()
    total_docs = sum(subj_counts.values()) if subj_counts else 0
    col_cnt.markdown(f"<div style='text-align:right; font-size:0.75rem; color:#94a3b8; padding-top:6px;'>{total_docs} Chunks</div>", unsafe_allow_html=True)

    st.divider()

    st.markdown("**Select Subject**")
    cur_subj = st.session_state.selected_subject

    # Global All Subjects button
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

    # Chat History
    st.markdown("**💬 Chat History**")
    saved_sessions = get_all_sessions()
    if saved_sessions:
        for s in saved_sessions:
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
                    st.rerun()

            if col_d.button("🗑️", key=f"del_{s_id}"):
                delete_session(s_id)
                if s_id == st.session_state.active_session_id:
                    switch_subject(cur_subj)
                st.rerun()
    else:
        st.caption("No saved chats yet.")

    st.divider()

    # AI Engine Selection
    st.markdown("**🧠 AI Engine**")
    groq_api_key = get_groq_api_key()
    ollama_online = is_ollama_online()
    local_models = get_local_ollama_models()

    engine_options = []
    if groq_api_key:
        engine_options.append("⚡ Auto (Groq Fast ➡️ Ollama Backup)")
        engine_options.append("🚀 Groq Cloud (llama-3.1-8b-instant)")
        engine_options.append("🚀 Groq Cloud (llama-3.3-70b-versatile)")
    if ollama_online:
        engine_options.append("💻 Ollama Local (Unlimited)")
    elif not groq_api_key:
        engine_options.append("⚡ Auto (Groq Cloud)")

    selected_engine = st.selectbox("AI Engine", engine_options, index=0, label_visibility="collapsed")
    selected_ollama_model = "llama3.2:latest"
    if "Ollama" in selected_engine or "Auto" in selected_engine:
        if local_models:
            selected_ollama_model = st.selectbox("Local Model", local_models, index=0)

    # Helper input for Groq API key if not configured in secrets/env
    if not groq_api_key and not ollama_online:
        st.warning("⚠️ **Groq API Key Required**")
        user_key_input = st.text_input("Enter Groq API Key", type="password", placeholder="gsk_...", key="user_groq_key")
        if user_key_input:
            os.environ["GROQ_API_KEY"] = user_key_input.strip()
            st.rerun()


# 4. MAIN CHAT AREA
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

# Render conversation history
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "✨"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Empty chat suggestions
if len(st.session_state.messages) == 0:
    st.markdown(f"### 💡 Studying **{subj_title}**")
    if cur_subj != "All Subjects":
        st.markdown(
            f"All your questions will be answered strictly from verified **{subj_title}** course notes and lab materials."
        )
    else:
        st.markdown(
            "Global Search mode: Questions are searched across all engineering subject notes."
        )

    st.markdown("---")
    st.markdown("#### 💬 Frequently Asked Questions:")
    sample_qs = SUBJECT_SAMPLE_QUESTIONS.get(cur_subj, SUBJECT_SAMPLE_QUESTIONS["All Subjects"])

    col1, col2 = st.columns(2)
    cols = [col1, col2, col1, col2]
    for i, q in enumerate(sample_qs[:4]):
        if cols[i].button(f"💬 {q}", key=f"sample_q_{i}", use_container_width=True):
            st.session_state.prompt_input = q
            st.rerun()


# 5. INPUT & STREAMING GENERATION
input_text = st.session_state.pop("prompt_input", None)
user_query = st.chat_input(f"Ask anything about {subj_title}...") or input_text

if user_query and user_query.strip():
    user_query = user_query.strip()
    if len(st.session_state.messages) == 0:
        st.session_state.session_title = user_query[:28]

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)

    recent_turns = st.session_state.messages[-7:-1]
    history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in recent_turns])

    with st.chat_message("assistant", avatar="✨"):
        try:
            with st.spinner("Searching course notes..."):
                context_str = retrieve_context(user_query, subject_filter=cur_subj, k=8)

            # Pre-LLM Relevance & Anti-Hallucination Gate
            is_relevant, out_of_scope_msg = is_context_relevant(
                query=user_query,
                context=context_str,
                active_subject=cur_subj
            )

            if not is_relevant:
                # Direct grounded response — preventing LLM hallucination
                st.markdown(out_of_scope_msg)
                st.session_state.messages.append({"role": "assistant", "content": out_of_scope_msg})
                save_session(
                    st.session_state.active_session_id,
                    st.session_state.session_title,
                    st.session_state.messages,
                    subject=cur_subj,
                )
            else:
                response_container = st.empty()
                full_response = ""

                def handle_fallback():
                    st.caption("⚡ *Groq cloud notice — switching engine.*")

                # Stream response through unified client (handles connection safety internally)
                for chunk in stream_llm_response(
                    active_subject=cur_subj,
                    context=context_str,
                    question=user_query,
                    chat_history=history_str,
                    selected_engine=selected_engine,
                    local_model=selected_ollama_model,
                    on_fallback=handle_fallback
                ):
                    full_response += chunk
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