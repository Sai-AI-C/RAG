"""
OmniDoc-RAG Ingestion Wrapper
Maintains backward compatibility by dispatching to src.ingestion.loader
"""

import os
from src.ingestion.loader import (
    load_single_pdf,
    load_single_docx,
    load_single_pptx,
    process_directory_incrementally,
)
from src.vectordb.vector_store import VectorStoreManager, get_vector_store
from src.embeddings.embedder import EmbeddingModel, get_embedding_model
from src.chunking.chunker import DocumentChunker, split_documents

# Backward compatible class names
EmbeddingM = EmbeddingModel
VectoreS = VectorStoreManager


def process_documents_incrementally(root_path: str = "./PDF_Data"):
    """Run incremental document ingestion."""
    from src.ingestion.loader import process_directory_incrementally as _process
    _process(root_path=root_path)


if __name__ == "__main__":
    process_documents_incrementally("./PDF_Data")
