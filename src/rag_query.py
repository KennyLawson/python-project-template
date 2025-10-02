from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from pathlib import Path
from .llm_lokal import generate_llm_reply

CHROMA_DIR = Path("models/vectorstore")
EMB_MODEL = "BAAI/bge-m3"

SYSTEM = (
    "Kamu Asisten Desa. Utamakan jawaban dari 'Konteks'. "
    "Bila tidak ada konteks relevan, katakan 'Saya belum punya datanya' dan sarankan langkah lanjut."
)

embedder = SentenceTransformer(EMB_MODEL)
client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings())
col = client.get_collection("desa_kb")


def retrieve_context(query: str, k: int = 6) -> str:
    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()[0]
    res = col.query(query_embeddings=[q_emb], n_results=k)
    docs = res.get("documents", [[]])[0]
    return "\n\n".join(docs[:3])  # ambil 3 teratas


def ask_with_rag(question: str) -> str:
    ctx = retrieve_context(question)
    prompt = (
        f"{SYSTEM}\n\nKonteks:\n{ctx}\n\n"
        f"Pertanyaan:\n{question}\n\n"
        "Instruksi: Jawab ≤120 kata, gunakan contoh langkah praktis bila perlu."
    )
    return generate_llm_reply(prompt)
