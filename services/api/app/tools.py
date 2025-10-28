import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from langchain_community.embeddings import OllamaEmbeddings


QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "local_wren_docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")


embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)




def ensure_collection(vector_size: int = 768):
    collections = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in collections:
        client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )




def query(q: str, k: int = 5):
    vec = embeddings.embed_query(q)
    res = client.search(collection_name=QDRANT_COLLECTION, query_vector=vec, limit=k)
    docs = []
    for p in res:
        payload = p.payload or {}
        docs.append({
        "text": payload.get("text", ""),
        "source": payload.get("source", "unknown"),
        "score": p.score,
        })
    return docs