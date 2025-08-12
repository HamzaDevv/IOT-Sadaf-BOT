import chromadb
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import logging
import requests
from config import OLLAMA_EMBED_URL, OLLAMA_EMBED_MODEL

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_ollama_embedding(text: str) -> List[float]:
    """Fetch embedding from local Ollama."""
    try:
        resp = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]
    except Exception as e:
        logger.error(f"Ollama embedding error: {e}")
        raise


class ChromaStore:
    """Handles persistent fact storage and retrieval with ChromaDB."""

    def __init__(self, collection_name: str, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a document with Ollama embeddings."""
        doc_id = str(uuid.uuid4())
        metadata = metadata or {}
        metadata["timestamp"] = datetime.now().isoformat()

        try:
            embedding = get_ollama_embedding(text)
            self.collection.add(
                documents=[text],
                metadatas=[metadata],
                embeddings=[embedding],
                ids=[doc_id],
            )
            logger.info(f"Stored doc {doc_id} in '{self.collection.name}'")
            return doc_id
        except Exception as e:
            logger.error(f"Chroma add_document error: {e}")
            raise

    def get_relevant_info(self, query: str, n_results: int = 5) -> str:
        """Retrieve relevant stored facts."""
        try:
            query_emb = get_ollama_embedding(query)
            results = self.collection.query(
                query_embeddings=[query_emb],
                n_results=n_results,
            )
            if results["documents"]:
                return "\n".join(f"- {doc}" for doc in results["documents"][0])
            return ""
        except Exception as e:
            logger.error(f"Chroma query error: {e}")
            return ""
