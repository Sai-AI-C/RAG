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
    SUBJECT_SCOPE_CONTEXT,
    is_short_query,
    normalize_subject_name,
)


def _subject_scope_fallback(subject: str) -> str:
    """Provide minimal selected-subject evidence when its notes have no readable text."""
    if subject in SUBJECT_SCOPE_CONTEXT:
        return SUBJECT_SCOPE_CONTEXT[subject]

    metadata = SUBJECT_METADATA.get(subject, {})
    title = metadata.get("title", subject)
    kind = metadata.get("type", "course")
    return (
        f"Selected subject scope: {title}. This is the {kind.lower()} collection "
        "selected by the student. Use the subject title as the topic boundary and "
        "state clearly when the indexed notes do not provide further detail."
    )


def expand_query(query: str, active_subject: str = "All Subjects") -> str:
    """
    Intelligently expands student queries using active-subject context without cross-polluting.
    Handles subject-specific abbreviations, common typos, and subject/lab overview questions.
    """
    active_subject = normalize_subject_name(active_subject) or "All Subjects"
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
        ("MSF", ("msf", "management science", "what is management science", "define management science")):
            "Management Science and Finance management science decision making operations research financial management",
        ("POE", ("poe", "economics", "what is economics", "define economics", "principles of economics")):
            "Principles of Economics economic principles demand supply markets microeconomics macroeconomics",
        ("CNS Lab", ("cns lab", "cns experiments", "list all cns lab experiments", "experiments")):
            "Cryptography and Network Security lab experiments practical programs Caesar cipher RSA DES AES hashing",
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

    subject_title = SUBJECT_METADATA.get(active_subject, {}).get("title", active_subject)
    if active_subject != "All Subjects" and re.search(
        r"\b(what is|define|explain|list|experiments|programs|overview)\b", cleaned
    ):
        return f"{raw} {subject_title}"

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
    subject = normalize_subject_name(subject)
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
    """
    Detects OCR artifact noise chunks with multiple heuristics:
    - Too short (single chars, page numbers, stray OCR fragments)
    - Contains unicode replacement chars (\ufffd = broken OCR character)
    - Very few real English words (word density)
    - Low alphanumeric ratio
    """
    if not text:
        return True
    cleaned = text.strip()

    # 1. Single char or tiny chunks (page numbers, stray marks)
    if len(cleaned) < 30:
        return True

    # 2. Unicode replacement char indicates broken OCR scan
    if cleaned.count('\ufffd') > 2:
        return True

    # 3. Low alphanumeric ratio
    alpha_chars = sum(c.isalnum() or c.isspace() for c in cleaned)
    ratio = alpha_chars / max(len(cleaned), 1)
    if ratio < 0.60:
        return True

    # 4. Very low real word density (garbled OCR has many non-word tokens)
    tokens = cleaned.split()
    if len(tokens) < 5:
        return True
    # Real word = at least 3 consecutive alphabetic chars
    real_words = sum(1 for t in tokens if sum(c.isalpha() for c in t) >= 3)
    word_ratio = real_words / max(len(tokens), 1)
    if word_ratio < 0.35:
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
    subject_filter = normalize_subject_name(subject_filter) or "All Subjects"
    search_query = expand_query(query, active_subject=subject_filter)
    embedder = get_embedding_model()
    db_manager = get_vector_store()

    # Generate dense query embedding (1D list of floats)
    query_embedding = embedder.encode_single(search_query)

    # Prepare subject filter where clause
    where_clause = None
    if subject_filter and subject_filter != "All Subjects":
        targets = get_related_subjects(subject_filter)
        # Lab lists must stay in the lab collection instead of returning theory chunks.
        is_lab_subject = SUBJECT_METADATA.get(subject_filter, {}).get("type") == "Lab"
        if is_lab_subject and re.search(r"\b(list|experiment|experiments|programs|practical)\b", query.lower()):
            targets = [subject_filter]
        if len(targets) == 1:
            where_clause = {"subject": targets[0]}
        elif len(targets) > 1:
            where_clause = {"subject": {"$in": targets}}

    # 1. Semantic search — fetch k+6 to compensate for garbled chunks that will be filtered
    sem_docs = db_manager.query_similarity(
        query_embedding=query_embedding,
        n_results=k + 6,
        where=where_clause
    )

    # 2. Safe keyword search (ONLY for non-abbreviation queries with length >= 4)
    # Short substrings like "DA" or "CN" match random substrings inside words and OCR noise!
    kw_docs = []
    raw_query = query.strip()
    if len(raw_query) >= 4 and not is_short_query(raw_query):
        keyword_queries = [raw_query]
        subject_keyword_queries = [
            SUBJECT_METADATA.get(subject_filter, {}).get("title", subject_filter),
        ]
        keyword_queries.extend(subject_keyword_queries)

        for keyword_query in keyword_queries:
            try:
                kw_docs.extend(db_manager.query_similarity(
                    query_embedding=query_embedding,
                    n_results=min(k, 4),
                    where=where_clause,
                    where_document={"$contains": keyword_query}
                ))
            except Exception:
                continue

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
        if re.search(r"\b(what is|define|explain|overview)\b", query.lower()):
            return _subject_scope_fallback(subject_filter)
        return ""

    merged_context = "\n\n---\n\n".join(merged_docs[:k])
    if subject_filter != "All Subjects" and re.search(
        r"\b(what is|define|explain)\b", query.lower()
    ):
        title_tokens = re.findall(
            r"\b[a-zA-Z0-9]{4,}\b",
            SUBJECT_METADATA.get(subject_filter, {}).get("title", subject_filter).lower(),
        )
        topic_tokens = tuple(title_tokens)
        if not any(token in merged_context.lower() for token in topic_tokens):
            return _subject_scope_fallback(subject_filter)

    return merged_context


def is_context_relevant(query: str, context: str, active_subject: str) -> Tuple[bool, Optional[str]]:
    """
    Pre-LLM Relevance & Anti-Hallucination Gate.
    Verifies if the retrieved context actually covers the requested concept for the active subject.
    If completely out-of-scope (e.g. asking "Data Analysis" in "Java Programming"), returns a clear
    subject-specific guidance message rather than letting the LLM hallucinate general knowledge.
    """
    active_subject = normalize_subject_name(active_subject) or active_subject

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

    # Check if the query is about the subject itself or a lab overview before enforcing
    # the short-context rejection guard. This avoids false negatives like "What is Economics?"
    # under POE when the retrieved snippet is a short but relevant definition.
    subj_tokens = set(re.findall(r'\b[a-zA-Z0-9_-]+\b', f"{active_subject} {subj_title}".lower()))
    common_lab_terms = {"lab", "experiment", "experiments", "manual", "syllabus", "overview", "programs", "list"}
    subject_self_query = any(
        w in subj_tokens or w in common_lab_terms or 
        any(w.startswith(t) or t.startswith(w) for t in subj_tokens if len(t) > 3)
        for w in query_keywords
    )
    context_subject_evidence = any(
        re.search(r"\b" + re.escape(token) + r"\b", context.lower())
        for token in subj_tokens
        if len(token) > 3
    )
    if subject_self_query and context_subject_evidence and len(context.strip()) > 25:
        return True, None

    if not context or len(context.strip()) < 50:
        fallback = OUT_OF_SCOPE_TEMPLATE.format(query=query, subject_title=subj_title)
        return False, fallback

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
