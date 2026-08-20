import os
import uuid
import zipfile
from typing import List, Set, Dict, Any, Optional
import chromadb
import numpy as np
from langchain_core.documents import Document
from src.utils.helpers import load_app_config

_VECTOR_STORE_INSTANCE = None


def ensure_vector_db_ready(persist_dir: str = "./pdf_db/chromadb", zip_path: str = "./PDF_db.zip") -> bool:
    """
    Ensure the vector database exists locally. If absent, attempts extraction from local zip
    or download from cloud URL (for Streamlit Cloud deployments).
    """
    if os.path.exists(persist_dir) and any(os.scandir(persist_dir)):
        return True

    # Attempt 1: Extract from local zip if present
    if os.path.exists(zip_path):
        print(f"📦 Unzipping local database package from {zip_path}...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(".")
            if os.path.exists(persist_dir):
                print("✅ Vector database extracted successfully.")
                return True
        except Exception as e:
            print(f"Extraction error: {e}")

    # Attempt 2: Download from VECTOR_DB_URL if defined (Streamlit Cloud secret or env)
    cloud_url = os.getenv("VECTOR_DB_URL", "")
    try:
        import streamlit as st
        if not cloud_url and "VECTOR_DB_URL" in st.secrets:
            cloud_url = st.secrets["VECTOR_DB_URL"]
    except Exception:
        pass

    if cloud_url:
        print("🌐 Downloading knowledge base from cloud storage...")
        try:
            if "drive.google.com" in cloud_url:
                import gdown
                gdown.download(cloud_url, zip_path, quiet=False, fuzzy=True)
            else:
                import requests
                resp = requests.get(cloud_url, stream=True, timeout=120)
                with open(zip_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(".")
                return True
        except Exception as e:
            print(f"Cloud download failed: {e}")

    return False


class VectorStoreManager:
    """Manages ChromaDB persistent client, collections, queries, and document additions."""

    def __init__(self, persist_dir: str = "./pdf_db/chromadb", collection_name: str = "Document__C"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize()

    def _initialize(self):
        try:
            ensure_vector_db_ready(self.persist_dir)
            os.makedirs(self.persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF embedding document for RAG"}
            )
            print(f"ChromaDB ready: collection='{self.collection_name}', items={self.collection.count()}")
        except Exception as e:
            print(f"Error initializing ChromaDB at {self.persist_dir}: {e}")
            raise

    def get_indexed_sources(self) -> Set[str]:
        """Fetch all unique source file paths currently stored in the collection."""
        try:
            indexed_sources = set()
            offset = 0
            page_size = 1000
            while True:
                results = self.collection.get(
                    include=["metadatas"],
                    limit=page_size,
                    offset=offset,
                )
                metadatas = results.get("metadatas", [])
                indexed_sources.update(
                    m.get("source") for m in metadatas if m and "source" in m
                )
                if len(metadatas) < page_size:
                    break
                offset += page_size
            return indexed_sources
        except Exception as e:
            print(f"Warning: Could not fetch indexed sources: {e}")
            return set()

    def delete_records(self, subject: Optional[str] = None) -> int:
        """Delete indexed records only; source documents on disk are never touched."""
        try:
            deleted = 0
            offset = 0
            page_size = 1000
            while True:
                filters = {"subject": subject} if subject else None
                results = self.collection.get(
                    where=filters,
                    include=[],
                    limit=page_size,
                    offset=offset,
                )
                ids = results.get("ids", [])
                if not ids:
                    break
                self.collection.delete(ids=ids)
                deleted += len(ids)
                if len(ids) < page_size:
                    break
            return deleted
        except Exception as e:
            raise RuntimeError(f"Could not delete vector records safely: {e}") from e

    def add_documents(self, documents: List[Document], embeddings: np.ndarray, batch_size: int = 500):
        """Add documents and embeddings in batches."""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        if not documents:
            return

        for start in range(0, len(documents), batch_size):
            end = min(start + batch_size, len(documents))
            batch_docs = documents[start:end]
            batch_embeddings = embeddings[start:end]
            ids, metadatas, doc_texts, embedding_list = [], [], [], []

            for i, (doc, embed) in enumerate(zip(batch_docs, batch_embeddings), start=start):
                doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
                ids.append(doc_id)
                meta = dict(doc.metadata or {})
                meta["content_length"] = len(doc.page_content)
                metadatas.append(meta)
                doc_texts.append(doc.page_content)
                embedding_list.append(embed.tolist() if hasattr(embed, "tolist") else list(embed))

            try:
                self.collection.add(
                    ids=ids,
                    embeddings=embedding_list,
                    metadatas=metadatas,
                    documents=doc_texts
                )
            except Exception as e:
                print(f"Error adding batch to vector store: {e}")

    def query_similarity(
        self,
        query_embedding: Any,
        n_results: int = 8,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Query documents by vector similarity with dimensional safety."""
        try:
            # Flatten or format query_embeddings to ChromaDB's expected List[List[float]]
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            if isinstance(query_embedding, list):
                if len(query_embedding) > 0 and isinstance(query_embedding[0], list):
                    # Already 2D list: [[0.1, ...]]
                    if len(query_embedding[0]) > 0 and isinstance(query_embedding[0][0], list):
                        # Was 3D list: [[[0.1, ...]]] -> take first 2D element
                        formatted_embeddings = query_embedding[0]
                    else:
                        formatted_embeddings = query_embedding
                else:
                    # 1D list: [0.1, ...] -> wrap to 2D: [[0.1, ...]]
                    formatted_embeddings = [query_embedding]
            else:
                formatted_embeddings = [[float(x) for x in query_embedding]]

            kwargs = {
                "query_embeddings": formatted_embeddings,
                "n_results": n_results,
            }
            if where:
                kwargs["where"] = where
            if where_document:
                kwargs["where_document"] = where_document

            results = self.collection.query(**kwargs)
            docs = results.get("documents", [[]])[0]
            return docs
        except Exception as e:
            print(f"Error querying vector store: {e}")
            return []


def get_vector_store() -> VectorStoreManager:
    """Singleton getter for VectorStoreManager."""
    global _VECTOR_STORE_INSTANCE
    if _VECTOR_STORE_INSTANCE is None:
        cfg = load_app_config()
        p_dir = cfg.get("database", {}).get("persist_directory", "./pdf_db/chromadb")
        c_name = cfg.get("database", {}).get("collection_name", "Document__C")
        _VECTOR_STORE_INSTANCE = VectorStoreManager(persist_dir=p_dir, collection_name=c_name)
    return _VECTOR_STORE_INSTANCE


def get_subject_counts() -> Dict[str, int]:
    """Return count of document chunks indexed for each subject."""
    manager = get_vector_store()
    try:
        results = manager.collection.get(include=["metadatas"])
        counts = {}
        for m in results.get("metadatas", []):
            if m and "subject" in m:
                subj = m["subject"]
                counts[subj] = counts.get(subj, 0) + 1
        return counts
    except Exception:
        return {}
