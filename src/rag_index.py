from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from pathlib import Path

KB_DIR = Path("data/kb")
CHROMA_DIR = Path("models/vectorstore")
EMB_MODEL = "BAAI/bge-m3"  # multilingual embeddings

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(
        anonymized_telemetry=False,  # <— matikan telemetry
    ),
)


def load_docs():
    loaders = [
        DirectoryLoader(
            "data/kb",
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8", "autodetect_encoding": False},
            show_progress=True,
        ),
        DirectoryLoader(
            "data/kb",
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8", "autodetect_encoding": False},
            show_progress=True,
        ),
        DirectoryLoader(
            "data/kb", glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
        ),
    ]
    docs = []
    for L in loaders:
        docs.extend(L.load())
    return docs


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)


def main():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    docs = load_docs()
    chunks = split_docs(docs)
    print(f"Loaded {len(docs)} docs → {len(chunks)} chunks")

    embedder = SentenceTransformer(EMB_MODEL)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
    )
    try:
        client.delete_collection("desa_kb")
    except Exception:
        pass
    col = client.create_collection("desa_kb", metadata={"hnsw:space": "cosine"})

    texts = [c.page_content for c in chunks]
    metas = [c.metadata for c in chunks]
    embs = embedder.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )
    ids = [f"doc-{i}" for i in range(len(texts))]

    col.add(documents=texts, embeddings=embs.tolist(), metadatas=metas, ids=ids)
    print(f"Indexed into {CHROMA_DIR}/ desa_kb")


if __name__ == "__main__":
    main()
