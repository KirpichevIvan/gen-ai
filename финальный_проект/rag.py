from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+(?:[-'][A-Za-zА-Яа-яЁё0-9]+)?")
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
ARTICLES_DIR = INPUT_DIR / "articles"
INDEX_PATH = INPUT_DIR / "local_bm25_index.json"


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    title: str
    url: str
    text: str


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower().replace("ё", "е") for m in TOKEN_RE.finditer(text)]


def read_article(path: Path) -> tuple[str, str, str]:
    text = path.read_text(encoding="utf-8").strip()
    title = next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")), path.stem)
    url = ""
    for line in text.splitlines():
        if line.startswith("Источник:"):
            url = line.split(":", 1)[1].strip()
            break
    return title, url, text


def split_text(text: str, size: int = 1200, overlap: int = 180) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= size:
            current = f"{current}\n\n{paragraph}".strip()
            continue
        if current:
            chunks.append(current)
        tail = current[-overlap:] if current and overlap else ""
        current = f"{tail}\n\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return chunks


def load_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(ARTICLES_DIR.glob("habr_*.txt")):
        title, url, text = read_article(path)
        source = "_".join(path.stem.split("_")[:2])
        for idx, chunk_text in enumerate(split_text(text)):
            chunks.append(
                Chunk(
                    id=f"{source}__{idx:03d}",
                    source=source,
                    title=title,
                    url=url,
                    text=chunk_text,
                )
            )
    return chunks


def build_index() -> dict:
    chunks = load_chunks()
    search_texts = [f"{c.title}\n{c.title}\n{c.text}" for c in chunks]
    data = {
        "ids": [c.id for c in chunks],
        "sources": [c.source for c in chunks],
        "titles": [c.title for c in chunks],
        "urls": [c.url for c in chunks],
        "texts": [c.text for c in chunks],
        "tokens": [tokenize(text) for text in search_texts],
    }
    INDEX_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def load_index() -> dict:
    if not INDEX_PATH.exists():
        return build_index()
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def search(query: str, k: int = 5) -> list[dict]:
    index = load_index()
    bm25 = BM25Okapi(index["tokens"])
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:k]
    rows = []
    max_score = max((score for _, score in ranked), default=0.0) or 1.0
    for pos, (idx, score) in enumerate(ranked, start=1):
        confidence = 1 / (1 + math.exp(-(float(score) / max_score * 6 - 3)))
        rows.append(
            {
                "rank": pos,
                "id": index["ids"][idx],
                "source": index["sources"][idx],
                "title": index["titles"][idx],
                "url": index["urls"][idx],
                "score": float(score),
                "confidence": round(confidence, 3),
                "text": index["texts"][idx],
            }
        )
    return rows


def _limit_words(text: str, max_words: int = 76) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def best_quote(text: str, query: str, max_chars: int = 430) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", compact)
    q_tokens = set(tokenize(query))
    if not sentences:
        return compact[:max_chars]
    best = max(sentences, key=lambda sent: len(q_tokens & set(tokenize(sent))))
    start = compact.find(best)
    if start < 0:
        return best[:max_chars].strip()
    half = max_chars // 2
    left = max(0, start - half)
    right = min(len(compact), start + len(best) + half)
    return _limit_words(compact[left:right].strip())


def answer_local(question: str, k: int = 5) -> dict:
    hits = search(question, k=k)
    evidence = [
        {
            "source": hit["source"],
            "quote": best_quote(hit["text"], question),
            "url": hit["url"],
        }
        for hit in hits[:3]
    ]
    source_list = ", ".join(dict.fromkeys(hit["source"] for hit in hits[:3]))
    answer = (
        "По локальному корпусу Habr наиболее релевантны источники "
        f"{source_list}. Короткий извлекательный ответ: "
        + " ".join(item["quote"] for item in evidence[:3])
    )
    return {"answer": answer, "evidence": evidence, "hits": hits}


def stats() -> dict:
    chunks = load_chunks()
    return {
        "articles": len(list(ARTICLES_DIR.glob("habr_*.txt"))),
        "chunks": len(chunks),
        "chars": sum(len(c.text) for c in chunks),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest")
    sub.add_parser("stats")
    ask = sub.add_parser("search")
    ask.add_argument("query")
    ask.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    if args.cmd == "ingest":
        data = build_index()
        print(json.dumps({"chunks": len(data["ids"]), "index": str(INDEX_PATH)}, ensure_ascii=False, indent=2))
    elif args.cmd == "stats":
        print(json.dumps(stats(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(search(args.query, k=args.k), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
