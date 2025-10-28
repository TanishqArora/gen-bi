import os, glob
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from langchain_community.embeddings import OllamaEmbeddings
from pypdf import PdfReader
from docx import Document


QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "local_wren_docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")


client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
embedder = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)




def load_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        text = []
        r = PdfReader(path)
        for p in r.pages:
            text.append(p.extract_text() or "")
            return "\n".join(text)
    if path.lower().endswith(".docx"):
        d = Document(path)
        return "\n".join(p.text for p in d.paragraphs)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()




def chunk(text: str, size=800, overlap=120):
    out = []
    i = 0
    while i < len(text):
        out.append(text[i:i+size])
        i += size - overlap
    return [c for c in out if c.strip()]




def upsert(points):
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)




def main():
    files = []
    for ext in ("*.txt", "*.md", "*.pdf", "*.docx"):
        files.extend(glob.glob(os.path.join("/data", ext)))
    pid = 0
    for fp in files:
        text = load_text(fp)
        for ch in chunk(text):
            vec = embedder.embed_query(ch)
            pt = PointStruct(id=None, vector=vec, payload={"text": ch, "source": os.path.basename(fp)})
            upsert([pt])
            pid += 1
    print(f"Indexed {pid} chunks from {len(files)} files.")


if __name__ == "__main__":
    main()