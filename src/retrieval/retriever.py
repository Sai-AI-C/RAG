import re
from typing import List, Tuple, Optional, Set
from src.embeddings.embedder import get_embedding_model
from src.vectordb.vector_store import get_vector_store
from src.prompts.prompt_templates import OUT_OF_SCOPE_TEMPLATE
from src.utils.helpers import (
    load_app_config,
    SUBJECT_ABBREV,
    SUBJECT_NAME_ABBREV,
    SUBJECT_METADATA,
    is_short_query,
)


def expand_query(query: str, active_subject: str = "All Subjects") -> str:
    """
    Intelligently expands student queries using active-subject context without cross-polluting.
    Handles subject-specific abbreviations, common typos, and subject/lab overview questions.
    """
    raw = query.strip()
    cleaned = raw.lower()

    # 1. Common typo corrections
    typos = {
        "regreesion": "regression",
        "algorithem": "algorithm",
        "dimentionality": "dimensionality",
        "supervized": "supervised",
        "unsupervized": "unsupervised",
        "classifcation": "classification",
    }
    for wrong, right in typos.items():
        cleaned = cleaned.replace(wrong, right)

    # 2. Check active subject's internal abbreviation dictionary first
    subj_abbrevs = SUBJECT_ABBREV.get(active_subject, {})
    for word in re.findall(r'\b[a-zA-Z0-9*]+\b', cleaned):
        if word in subj_abbrevs:
            expansion = subj_abbrevs[word]
            return f"{raw} ({expansion})"

    # 3. Subject-level direct question overrides
    subject_direct_overrides = {
        ("CN Notes", ("cn", "define cn", "what is cn", "explain cn")):
            "Computer Networks architecture OSI TCP/IP model layers protocols",
        ("CN Lab", ("cn", "define cn", "what is cn", "cn lab", "experiments")):
            "Computer Networks lab experiments socket programming networking",
        ("AI", ("ai", "define ai", "what is ai", "explain ai")):
            "Artificial Intelligence definitions agents turing test heuristic search",
        ("AI-NLP Lab", ("ai", "nlp", "ai lab", "nlp lab", "experiments")):
            "Artificial Intelligence NLP lab experiments programs",
        ("ML notes", ("ml", "define ml", "what is ml", "types ml", "types of ml")):
            "Machine Learning types supervised unsupervised reinforcement learning classification regression",
        ("ML Lab", ("ml lab", "experiments", "list of experiments", "ml programs")):
            "Machine Learning lab experiments algorithms dataset classification regression",
        ("DAA Notes", ("da", "daa", "define da", "what is da", "what is daa")):
            "Design and Analysis of Algorithms asymptotic notations time complexity divide and conquer",
        ("ACS Lab", ("acs", "define acs", "what is acs", "acs lab", "what is acs lab", "experiments", "list of experiments")):
            "Advanced Communication Systems lab experiments modulation transmission fiber optics microwave",
        ("Neural network and deep learning", ("nnd", "nndl", "types of nnd", "what is nnd")):
            "Neural Networks Deep Learning types feed-forward recurrent RNN CNN autoencoders",
        ("STM Notes", ("stm", "what is stm", "define stm")):
            "Software Testing Methodologies testing types coverage white box black box",
        ("DBMS", ("dbms", "what is dbms", "define dbms")):
            "Database Management Systems relational database schema SQL normalization",
        ("DBMS Lab", ("dbms lab", "experiments", "sql queries", "programs")):
            "Database Management Systems lab queries triggers normalization tables",
        ("OS", ("os", "what is os", "define os")):
            "Operating Systems process management scheduling memory virtualization",
        ("Java", ("java", "what is java", "oop java")):
            "Java Programming OOP concepts classes objects inheritance polymorphism exception handling",
        ("Java Lab", ("java lab", "programs", "experiments")):
            "Java Programming lab experiments programs multithreading socket file handling",
    }

    for (target_subj, triggers), expanded_text in subject_direct_overrides.items():
        if active_subject == target_subj:
            for trigger in triggers:
                if trigger == cleaned or trigger in cleaned:
                    return expanded_text

    # 4. If in "All Subjects" mode or global search, check cross-subject abbreviations
    if active_subject == "All Subjects":
        for word in cleaned.split():
            if word in SUBJECT_NAME_ABBREV:
                full_name = SUBJECT_NAME_ABBREV[word]
                return f"{raw} ({full_name})"

    return raw


def get_related_subjects(subject: str) -> List[str]:
    """
    Returns related subject labels for multi-collection/subject retrieval filtering.
    For instance, linking Lab and Theory notes together.
    """
    if not subject or subject == "All Subjects":
        return []

    related = [subject]
    if "Notes" in subject:
        related.append(subject.replace("Notes", "Lab").strip())
    elif "Lab" in subject:
        related.append(subject.replace("Lab", "Notes").strip())

    # Curricular pairings
    curriculum_links = {
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

    if subject in curriculum_links:
        related.extend(curriculum_links[subject])

    return list(set(related))


def is_garbled_ocr(text: str) -> bool:
    """Detects OCR artifact noise chunks with low alphanumeric or broken token density."""
    if not text or len(text.strip()) < 20:
        return True
    cleaned = text.strip()
    alpha_chars = sum(c.isalnum() or c.isspace() for c in cleaned)
    ratio = alpha_chars / max(len(cleaned), 1)
    if ratio < 0.65:
        return True
    return False


def retrieve_context(query: str, subject_filter: str = "All Subjects", k: int = 8) -> str:
    """
    Robust Hybrid RAG Retrieval:
    1. Expands query within active subject context.
    2. Performs dense semantic vector search via ChromaDB.
    3. For multi-word/long queries (>= 4 chars), performs exact keyword filtering.
    4. Filters out garbled OCR noise and deduplicates chunks.
    """
    search_query = expand_query(query, active_subject=subject_filter)
    embedder = get_embedding_model()
    db_manager = get_vector_store()

    # Generate dense query embedding (1D list of floats)
    query_embedding = embedder.encode_single(search_query)

    # Prepare subject filter where clause
    where_clause = None
    if subject_filter and subject_filter != "All Subjects":
        targets = get_related_subjects(subject_filter)
        if len(targets) == 1:
            where_clause = {"subject": targets[0]}
        elif len(targets) > 1:
            where_clause = {"subject": {"$in": targets}}

    # 1. Semantic search (primary)
    sem_docs = db_manager.query_similarity(
        query_embedding=query_embedding,
        n_results=k,
        where=where_clause
    )

    # 2. Safe keyword search (ONLY for non-abbreviation queries with length >= 4)
    # Short substrings like "DA" or "CN" match random substrings inside words and OCR noise!
    kw_docs = []
    raw_query = query.strip()
    if len(raw_query) >= 4 and not is_short_query(raw_query):
        try:
            kw_docs = db_manager.query_similarity(
                query_embedding=query_embedding,
                n_results=min(k, 4),
                where=where_clause,
                where_document={"$contains": raw_query}
            )
        except Exception:
            kw_docs = []

    # 3. Merge, filter out noisy OCR, and deduplicate
    seen_prefixes: Set[str] = set()
    merged_docs: List[str] = []

    for doc in (kw_docs + sem_docs):
        if not doc or not doc.strip():
            continue
        if is_garbled_ocr(doc):
            continue

        prefix = doc.strip()[:140]
        if prefix not in seen_prefixes:
            seen_prefixes.add(prefix)
            merged_docs.append(doc.strip())

    if not merged_docs:
        return ""

    return "\n\n---\n\n".join(merged_docs[:k])


def is_context_relevant(query: str, context: str, active_subject: str) -> Tuple[bool, Optional[str]]:
    """
    Pre-LLM Relevance & Anti-Hallucination Gate.
    Verifies if the retrieved context actually covers the requested concept for the active subject.
    If completely out-of-scope (e.g. asking "Data Analysis" in "Java Programming"), returns a clear
    subject-specific guidance message rather than letting the LLM hallucinate general knowledge.
    """
    if not context or len(context.strip()) < 50:
        subj_meta = SUBJECT_METADATA.get(active_subject, {})
        subj_title = subj_meta.get("title", active_subject)
        fallback = OUT_OF_SCOPE_TEMPLATE.format(query=query, subject_title=subj_title)
        return False, fallback

    if active_subject == "All Subjects":
        return True, None

    subj_meta = SUBJECT_METADATA.get(active_subject, {})
    subj_title = subj_meta.get("title", active_subject)

    # Extract meaningful keywords from query (excluding stop words)
    stop_words = {
        "what", "is", "the", "in", "of", "and", "or", "to", "a", "an", "explain",
        "describe", "define", "about", "give", "list", "types", "different",
        "how", "many", "does", "do", "for", "with", "from", "by", "on", "notes",
        "all", "some", "can", "you", "me", "tell", "show", "please"
    }
    words = re.findall(r'\b[a-zA-Z0-9_-]+\b', query.lower())
    query_keywords = [w for w in words if w not in stop_words and len(w) > 1]

    if not query_keywords:
        return True, None

    # Check if the query is asking about the subject itself or lab experiments
    subj_tokens = set(re.findall(r'\b[a-zA-Z0-9_-]+\b', f"{active_subject} {subj_title}".lower()))
    common_lab_terms = {"lab", "experiment", "experiments", "manual", "syllabus", "overview", "programs", "list"}
    if any(w in subj_tokens or w in common_lab_terms for w in query_keywords) and len(context.strip()) > 80:
        return True, None

    # Check if any significant query keywords appear in the retrieved context
    context_lower = context.lower()
    expanded = expand_query(query, active_subject=active_subject).lower()
    expanded_words = [w for w in re.findall(r'\b[a-zA-Z0-9_-]+\b', expanded) if w not in stop_words and len(w) > 1]

    match_count = sum(1 for kw in query_keywords if re.search(r'\b' + re.escape(kw) + r'\b', context_lower))
    expanded_match_count = sum(1 for kw in expanded_words if re.search(r'\b' + re.escape(kw) + r'\b', context_lower))

    # Cross-subject mismatch detection (e.g. asking "Data Analysis" in "Java")
    is_subject_mismatch = False
    if active_subject in ["Java", "Java Lab"] and "data analysis" in query.lower():
        is_subject_mismatch = True
    elif active_subject in ["CN Notes", "CN Lab"] and "normal distribution" in query.lower():
        is_subject_mismatch = True
    elif active_subject in ["DBMS", "DBMS Lab"] and "turing machine" in query.lower():
        is_subject_mismatch = True

    if is_subject_mismatch or (match_count == 0 and expanded_match_count == 0):
        fallback = OUT_OF_SCOPE_TEMPLATE.format(query=query, subject_title=subj_title)
        return False, fallback

    return True, None
