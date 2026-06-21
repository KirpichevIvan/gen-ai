"""RAG pipeline for the Seminar 4 homework corpus.

Commands:
    python pipeline.py stats
    python pipeline.py ingest --strategy recursive
    python pipeline.py retrieve "Что такое DCD?" --strategy recursive --k 5
    python pipeline.py ask "Как проверять groundedness RAG-ответа?"

The pipeline is intentionally usable without an LLM key: retrieval and an
extractive answer fallback work offline after ingest. If LLM_BASE_URL or
OPENAI_API_KEY is configured, `ask` can generate a structured answer with an
OpenAI-compatible chat model.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from schema import RAGAnswer


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
STARTER_DIR = BASE_DIR.parent / "starter"
BM25_CACHE_TEMPLATE = "bm25_cache_{strategy}.json"
COLLECTION_PREFIX = "habr_rag"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
VALID_STRATEGIES = ("fixed", "recursive")

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+(?:[-'][A-Za-zА-Яа-яЁё0-9]+)?")


@dataclass(frozen=True)
class Document:
    source: str
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    chunk_id: int
    text: str


def print_safe(text: str = "") -> None:
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding))


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower().replace("ё", "е") for m in TOKEN_RE.finditer(text)]


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(data_dir.glob("habr_*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        title = next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")), path.stem)
        documents.append(Document(source=path.stem, title=title, text=text))
    return documents


def chunk_fixed(text: str, chunk_size: int = 2000) -> list[str]:
    return [text[i : i + chunk_size].strip() for i in range(0, len(text), chunk_size) if text[i : i + chunk_size].strip()]


def chunk_recursive(text: str, chunk_size: int = 400, chunk_overlap: int = 80) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "],
    )
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


def get_chunker(strategy: str) -> Callable[[str], list[str]]:
    if strategy == "fixed":
        return chunk_fixed
    if strategy == "recursive":
        return chunk_recursive
    raise ValueError(f"Unknown chunking strategy: {strategy}")


def make_chunks(documents: list[Document], strategy: str) -> list[Chunk]:
    chunker = get_chunker(strategy)
    chunks: list[Chunk] = []
    for doc in documents:
        for index, chunk_text in enumerate(chunker(doc.text)):
            chunks.append(
                Chunk(
                    id=f"{doc.source}__{strategy}__{index}",
                    source=doc.source,
                    chunk_id=index,
                    text=chunk_text,
                )
            )
    return chunks


def get_embed_fn():
    from chromadb.utils import embedding_functions

    allow_download = os.environ.get("RAG_ALLOW_MODEL_DOWNLOAD", "").lower() in {"1", "true", "yes", "on"}
    if not allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        local_files_only=not allow_download,
    )


def get_collection(strategy: str):
    import chromadb

    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return chroma.get_or_create_collection(
        name=f"{COLLECTION_PREFIX}_{strategy}",
        embedding_function=get_embed_fn(),
        metadata={"hnsw:space": "cosine"},
    )


def get_collection_count(strategy: str) -> int:
    import chromadb

    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        return chroma.get_collection(name=f"{COLLECTION_PREFIX}_{strategy}").count()
    except Exception:
        return 0


def bm25_cache_path(strategy: str) -> Path:
    return BASE_DIR / BM25_CACHE_TEMPLATE.format(strategy=strategy)


def write_bm25_cache(chunks: list[Chunk], strategy: str) -> None:
    data = {
        "ids": [chunk.id for chunk in chunks],
        "sources": [chunk.source for chunk in chunks],
        "texts": [chunk.text for chunk in chunks],
        "tokens": [tokenize(chunk.text) for chunk in chunks],
    }
    bm25_cache_path(strategy).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def ingest(strategy: str = "recursive") -> None:
    documents = load_documents()
    if not documents:
        raise RuntimeError(f"No Habr .txt documents found in {DATA_DIR}")

    chunks = make_chunks(documents, strategy)
    if not chunks:
        raise RuntimeError("No chunks produced")

    collection = get_collection(strategy)
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    batch_size = 128
    start = time.time()
    for start_index in range(0, len(chunks), batch_size):
        batch = chunks[start_index : start_index + batch_size]
        collection.add(
            ids=[chunk.id for chunk in batch],
            documents=[chunk.text for chunk in batch],
            metadatas=[
                {"source": chunk.source, "chunk_id": chunk.chunk_id, "strategy": strategy}
                for chunk in batch
            ],
        )

    write_bm25_cache(chunks, strategy)

    print_safe(f"Indexed strategy: {strategy}")
    print_safe(f"Documents: {len(documents)}")
    print_safe(f"Chunks: {len(chunks)}")
    print_safe(f"Collection: {COLLECTION_PREFIX}_{strategy}")
    print_safe(f"Chroma path: {CHROMA_DIR}")
    print_safe(f"BM25 cache: {bm25_cache_path(strategy).name}")
    print_safe(f"Done in {time.time() - start:.1f}s")


def retrieve(query: str, strategy: str = "recursive", k: int = 5) -> dict:
    collection = get_collection(strategy)
    if collection.count() == 0:
        raise RuntimeError(f"Collection for strategy '{strategy}' is empty. Run: python pipeline.py ingest --strategy {strategy}")
    return collection.query(query_texts=[query], n_results=k)


def result_rows(hits: dict) -> list[dict]:
    rows: list[dict] = []
    ids = hits.get("ids", [[]])[0]
    docs = hits.get("documents", [[]])[0]
    metas = hits.get("metadatas", [[]])[0]
    distances = hits.get("distances", [[]])[0] if hits.get("distances") else [None] * len(ids)
    for cid, text, meta, distance in zip(ids, docs, metas, distances):
        rows.append(
            {
                "id": cid,
                "source": meta.get("source") if meta else cid.split("__")[0],
                "chunk_id": meta.get("chunk_id") if meta else None,
                "distance": distance,
                "text": text,
            }
        )
    return rows


def build_prompt(question: str, hits: dict) -> str:
    context_blocks = []
    for row in result_rows(hits):
        context_blocks.append(f"[{row['id']}]\n{row['text']}")
    context = "\n\n---\n\n".join(context_blocks)
    return (
        "Ты отвечаешь на вопрос по корпусу статей Хабра про RAG. "
        "Используй только контекст ниже. Если ответа нет в контексте, честно скажи, что данных недостаточно.\n\n"
        "Требования к JSON:\n"
        "- answer: краткий ответ на русском;\n"
        "- quotes: 1-5 коротких точных цитат из контекста;\n"
        "- sources: id чанков, из которых взяты факты;\n"
        "- confidence: число от 0 до 1.\n\n"
        f"Контекст:\n{context}\n\n"
        f"Вопрос: {question}"
    )


def extractive_answer(question: str, hits: dict) -> RAGAnswer:
    rows = result_rows(hits)
    if not rows:
        return RAGAnswer(answer="В корпусе не найден релевантный контекст.", confidence=0.0, quotes=[], sources=[])

    snippets = []
    for row in rows[:3]:
        text = re.sub(r"\s+", " ", row["text"]).strip()
        snippets.append(text[:420] + ("..." if len(text) > 420 else ""))

    answer = (
        "LLM-клиент не настроен, поэтому возвращаю извлекательный ответ по найденным чанкам. "
        "Самые релевантные фрагменты указывают на следующие источники: "
        + ", ".join(row["source"] for row in rows[:3])
        + "."
    )
    return RAGAnswer(
        answer=answer,
        quotes=snippets,
        sources=[row["id"] for row in rows[:5]],
        confidence=0.55,
    )


def can_use_llm() -> bool:
    return bool(os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_API_KEY"))


def llm_answer(question: str, hits: dict) -> RAGAnswer:
    if STARTER_DIR.exists() and str(STARTER_DIR) not in sys.path:
        sys.path.insert(0, str(STARTER_DIR))

    from llm_client import get_model, make_client

    client = make_client()
    prompt = build_prompt(question, hits)
    return client.chat.completions.create(
        model=get_model(),
        response_model=RAGAnswer,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_retries=1,
    )


def ask(question: str, strategy: str = "recursive", k: int = 5, use_llm: bool | None = None) -> RAGAnswer:
    hits = retrieve(question, strategy=strategy, k=k)
    if use_llm is None:
        use_llm = can_use_llm()
    if use_llm:
        return llm_answer(question, hits)
    return extractive_answer(question, hits)


def show_retrieve(query: str, strategy: str, k: int) -> None:
    hits = retrieve(query, strategy=strategy, k=k)
    for rank, row in enumerate(result_rows(hits), start=1):
        distance = row["distance"]
        distance_text = f"{distance:.4f}" if isinstance(distance, float) else "n/a"
        preview = re.sub(r"\s+", " ", row["text"]).strip()
        print_safe(f"{rank}. {row['id']} | distance={distance_text}")
        print_safe(f"   {preview[:260]}{'...' if len(preview) > 260 else ''}")


def show_stats() -> None:
    documents = load_documents()
    print_safe(f"Documents: {len(documents)}")
    print_safe(f"Total chars: {sum(len(doc.text) for doc in documents)}")
    for strategy in VALID_STRATEGIES:
        chunks = make_chunks(documents, strategy)
        cache = bm25_cache_path(strategy)
        print_safe(
            f"{strategy}: chunks_if_ingested={len(chunks)}, bm25_cache={cache.exists()}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Homework RAG pipeline for Habr corpus")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stats_parser = subparsers.add_parser("stats")
    stats_parser.set_defaults(func=lambda args: show_stats())

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("--strategy", choices=VALID_STRATEGIES, default="recursive")
    ingest_parser.set_defaults(func=lambda args: ingest(strategy=args.strategy))

    retrieve_parser = subparsers.add_parser("retrieve")
    retrieve_parser.add_argument("query")
    retrieve_parser.add_argument("--strategy", choices=VALID_STRATEGIES, default="recursive")
    retrieve_parser.add_argument("--k", type=int, default=5)
    retrieve_parser.set_defaults(func=lambda args: show_retrieve(args.query, args.strategy, args.k))

    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--strategy", choices=VALID_STRATEGIES, default="recursive")
    ask_parser.add_argument("--k", type=int, default=5)
    llm_group = ask_parser.add_mutually_exclusive_group()
    llm_group.add_argument("--llm", action="store_true", help="Force LLM generation")
    llm_group.add_argument("--no-llm", action="store_true", help="Force extractive fallback")
    ask_parser.set_defaults(func=run_ask_command)

    return parser.parse_args()


def run_ask_command(args: argparse.Namespace) -> None:
    use_llm = True if args.llm else False if args.no_llm else None
    answer = ask(args.question, strategy=args.strategy, k=args.k, use_llm=use_llm)
    print_safe(answer.model_dump_json(indent=2))


def main() -> int:
    args = parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
