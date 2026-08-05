from ollama import embeddings
from importlib.metadata import metadata
from annotated_types import DocInfo
import os, uuid
from typing import List 
import chromadb
import numpy as np
from langchain_community.document_loaders import DirectoryLoader,PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

def load_all_documents(path: str )->List[Document]:
    loader=DirectoryLoader(path,glob="**/*.pdf",loader_cls=PyMuPDFLoader,show_progress=False)
    return loader.load()
class EmbeddingM:
    def __init__(self, model_N: str = "all-MiniLM-L6-v2"):
        self.model = None
        self.model_N = model_N
        self._load_model()

    def _load_model(self):
        try:
            self.model = SentenceTransformer(self.model_N)
            print(f"Model Loaded successfully. Embedding Model: {self.model_N}")
        except Exception as e:
            print(f"Error loading model {self.model_N}: {e}")
            raise

    def split_texts(self, docs: List[Document], chunk_S: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        text_S = RecursiveCharacterTextSplitter(chunk_size=chunk_S, chunk_overlap=chunk_overlap)
        split_docs: List[Document] = []
        for doc in docs:
            chunks = text_S.split_text(doc.page_content)
            for chunk in chunks:
                split_docs.append(Document(page_content=chunk, metadata={**doc.metadata, "source": doc.metadata.get("source", "unknown")}))
        print(f"Split {len(docs)} source documents into {len(split_docs)} chunks.")
        return split_docs

    def generate_E(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model is not loaded.")
        if isinstance(texts, str):
            texts = [texts]
        print(f"Generating embeddings for {len(texts)} texts...")
        e = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings shape: {e.shape}")
        return e
class VectoreS:
    def __init__(self, collection_N: str = "Document__C", persist_D: str = "./pdf_db/chromadb"):
        self.collection_N = collection_N
        self.persistent_D = persist_D
        self.client = None
        self.collection = None
        self._initialize_S()

    def _initialize_S(self):
        try:
            os.makedirs(self.persistent_D,exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persistent_D)
            self.collection=self.client.get_or_create_collection(name=self.collection_N,metadata={"description":"PDF embedding document for RAG"})
            print(f"Existing document count in collection : {self.collection.count()}")
        except Exception as e:
            print(f"Error while initialize the Vector Store : {e}")
            raise
    def add_Document(self,documents:List[Document],embeddings:np.ndarray,batch_size: int=1000):
        if len(documents)!=len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
            print(f"Adding {len(documents)} documents to vector store in batches of {batch_size}")
            add_C=0
        for start in range(0,len(documents),batch_size):
            end=min(start+batch_size,len(documents))
            batch_docs=documents[start:end]
            batch_embeddings=embeddings[start:end]
            ids=[]
            metadatas=[]
            doc_T=[]
            embedding_L=[]
            for i,(doc,embedd) in enumerate(zip(batch_docs,batch_embeddings),start=start):
                d_id=f"doc_{uuid.uuid4().hex[:8]}_{i}"
                ids.append(d_id)
                metadata=dict(doc.metadata or {})
                metadata["doc_index"]=i
                metadata["content_length"]=len(doc.page_content)
                metadatas.append(metadata)
                doc_T.append(doc.page_content)
                embedding_L.append(embedd.tolist())
            try:
                self.collection.add(ids=ids,embeddings=embedding_L,metadatas=metadatas,documents=doc_T)
                add_C=len(batch_docs)
                print(f"Succesfully added batch {start//batch_size+1} with {len(batch_docs)} documents")
            except Exception as e:
                print(f"Error adding documents to vector store in {start//batch_size+1}:{e}")
                raise
        print(f"Successfully added {add_C} documents to vector store ")
        print(f"Total document count collections is : {self.collection.count()}")   

if __name__=="__main__":
    embedding_M=EmbeddingM()
    pdf_docs=load_all_documents("./PDF_Data")
    chunks=embedding_M.split_texts(pdf_docs)
    embeddings=embedding_M.generate_E([chunk.page_content for chunk in chunks])
    vectors_store=VectoreS()
    vectors_store.add_Document(chunks,embeddings)