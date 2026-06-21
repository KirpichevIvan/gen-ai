from __future__ import annotations

import argparse
import html
import json
import os
import re
from pathlib import Path
from urllib.parse import quote_plus

import httpx
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
CACHE_PATH = BASE_DIR / "input" / "stackoverflow_cache.json"
API_URL = "https://api.stackexchange.com/2.3/search/advanced"
EXCERPTS_URL = "https://api.stackexchange.com/2.3/search/excerpts"
ANSWERS_URL = "https://api.stackexchange.com/2.3/answers/{ids}"
QUESTION_ANSWERS_URL = "https://api.stackexchange.com/2.3/questions/{ids}/answers"

load_dotenv(BASE_DIR / ".env")


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zа-яё0-9]+", text.lower()))


def _load_cache() -> list[dict]:
    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))


def _strip_html(value: str) -> str:
    value = re.sub(r"(?is)<pre><code>.*?</code></pre>", " ", value)
    value = re.sub(r"(?is)<code>(.*?)</code>", r"\1", value)
    value = re.sub(r"(?i)<br\s*/?>", "\n", value)
    value = re.sub(r"(?i)</p>", "\n", value)
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def _limit_words(text: str, max_words: int = 76) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _cache_search(query: str, k: int) -> list[dict]:
    q = _tokens(query)
    rows = []
    for item in _load_cache():
        haystack = f"{item['query']} {item['title']} {item['summary']}"
        overlap = len(q & _tokens(haystack))
        rows.append({**item, "source": "stackoverflow_cache", "overlap": overlap})
    rows.sort(key=lambda item: (item["overlap"], item.get("score", 0)), reverse=True)
    return rows[:k]


def _stackoverflow_query(query: str) -> str:
    lower = query.lower()
    expansions: list[str] = []
    dictionary = {
        "метрик": "RAG evaluation metrics context precision context recall faithfulness",
        "metrics": "Graph RAG Pipeline Metrics evaluate",
        "graph rag": "Graph RAG Pipeline Metrics evaluate",
        "оцен": "Graph RAG Pipeline Metrics evaluate",
        "качество retrieval": "retrieval quality",
        "качество generation": "generation quality",
        "цитат": "RAG citations retrieved documents page citations hallucinations",
        "page citations": "RAG citations retrieved documents page citations hallucinations",
        "retrieved documents": "RAG citations retrieved documents page citations hallucinations",
        "выдуман": "hallucinated citations",
        "галлюцинац": "hallucinations RAG groundedness",
        "чанк": "RecursiveCharacterTextSplitter chunk_size chunk_overlap LangChain",
        "chunk": "RecursiveCharacterTextSplitter chunk_size chunk_overlap LangChain",
        "chunk_size": "RecursiveCharacterTextSplitter chunk_size chunk_overlap LangChain",
        "chunk_overlap": "RecursiveCharacterTextSplitter chunk_size chunk_overlap LangChain",
        "recursivecharactertextsplitter": "RecursiveCharacterTextSplitter chunk_size chunk_overlap LangChain",
        "haystack": "RAG pipeline Haystack Milvus embedder",
        "milvus": "RAG pipeline Haystack Milvus embedder",
        "embedder": "RAG pipeline Haystack Milvus embedder",
        "pipeline": "RAG pipeline",
        "table": "BM25 vector search table RAG hybrid retrieval",
        "table-based": "BM25 vector search table RAG hybrid retrieval",
        "таблич": "BM25 vector search table RAG hybrid retrieval",
        "bm25": "BM25 vector search hybrid search RAG",
        "hybrid": "BM25 vector search hybrid search RAG",
        "qdrant": "Qdrant hybrid search dense sparse vectors",
        "dense": "dense sparse vectors hybrid search Qdrant",
        "sparse": "dense sparse vectors hybrid search Qdrant",
        "api": "Stack Exchange API Python quota backoff site stackoverflow",
        "python": "Python Stack Exchange API",
        "quota": "Stack Exchange API Python quota backoff site stackoverflow",
        "backoff": "Stack Exchange API Python quota backoff site stackoverflow",
        "langchain": "LangChain RetrievalQA return_source_documents source_documents",
        "retrievalqa": "LangChain RetrievalQA return_source_documents source_documents",
        "return_source_documents": "LangChain RetrievalQA return_source_documents source_documents",
        "source_documents": "LangChain RetrievalQA return_source_documents source_documents",
        "source": "LangChain RetrievalQA return_source_documents source_documents",
    }
    for marker, expansion in dictionary.items():
        if marker in lower:
            expansions.append(expansion)
    if expansions:
        return " ".join(dict.fromkeys(expansions))
    return query


def _live_excerpt_search(client: httpx.Client, query: str, k: int) -> list[dict]:
    response = client.get(
        EXCERPTS_URL,
        params={
            "site": "stackoverflow",
            "order": "desc",
            "sort": "relevance",
            "q": query,
            "pagesize": k,
            "filter": "default",
        },
    )
    response.raise_for_status()
    rows = []
    for item in response.json().get("items", []):
        title = html.unescape(item.get("title", ""))
        excerpt = _strip_html(item.get("excerpt", ""))
        question_id = item.get("question_id")
        answer_id = item.get("answer_id")
        if question_id:
            url = f"https://stackoverflow.com/questions/{question_id}"
        elif answer_id:
            url = f"https://stackoverflow.com/a/{answer_id}"
        else:
            url = f"https://stackoverflow.com/search?q={quote_plus(query)}"
        rows.append(
            {
                "query": query,
                "question_id": question_id,
                "answer_id": answer_id,
                "title": title,
                "url": url,
                "score": item.get("score", 0),
                "is_answered": item.get("item_type") == "answer",
                "summary": _limit_words(excerpt or title),
                "source": "stackoverflow_api_excerpt",
            }
        )
    return rows


def _fetch_answer_bodies(client: httpx.Client, questions: list[dict]) -> dict[int, str]:
    accepted_ids = [
        str(item["accepted_answer_id"])
        for item in questions
        if item.get("accepted_answer_id")
    ]
    bodies: dict[int, str] = {}
    if accepted_ids:
        response = client.get(
            ANSWERS_URL.format(ids=";".join(accepted_ids)),
            params={"site": "stackoverflow", "filter": "withbody"},
        )
        response.raise_for_status()
        for item in response.json().get("items", []):
            bodies[int(item["answer_id"])] = _strip_html(item.get("body", ""))

    missing_question_ids = [
        str(item["question_id"])
        for item in questions
        if not item.get("accepted_answer_id")
    ]
    if missing_question_ids:
        response = client.get(
            QUESTION_ANSWERS_URL.format(ids=";".join(missing_question_ids)),
            params={
                "site": "stackoverflow",
                "order": "desc",
                "sort": "votes",
                "pagesize": 1,
                "filter": "withbody",
            },
        )
        response.raise_for_status()
        for item in response.json().get("items", []):
            bodies[int(item["question_id"])] = _strip_html(item.get("body", ""))
    return bodies


def _live_search(query: str, k: int) -> list[dict]:
    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        excerpt_rows = _live_excerpt_search(client, query, k)
        if excerpt_rows:
            return excerpt_rows

        params = {
            "order": "desc",
            "sort": "relevance",
            "site": "stackoverflow",
            "q": query,
            "pagesize": k,
            "filter": "default",
        }
        response = client.get(API_URL, params=params)
        response.raise_for_status()
        payload = response.json()
        questions = payload.get("items", [])
        bodies = _fetch_answer_bodies(client, questions)

    rows = []
    for item in questions:
        title = html.unescape(item.get("title", ""))
        link = item.get("link") or f"https://stackoverflow.com/search?q={quote_plus(query)}"
        body = ""
        if item.get("accepted_answer_id"):
            body = bodies.get(int(item["accepted_answer_id"]), "")
        if not body:
            body = bodies.get(int(item.get("question_id", 0)), "")
        summary = _limit_words(body if body else title)
        rows.append(
            {
                "query": query,
                "question_id": item.get("question_id"),
                "title": title,
                "url": link,
                "score": item.get("score", 0),
                "is_answered": bool(item.get("is_answered")),
                "summary": summary,
                "source": "stackoverflow_api",
            }
        )
    return rows


def search_stackoverflow(query: str, k: int = 3) -> dict:
    api_query = _stackoverflow_query(query)
    mode = os.environ.get("STACKOVERFLOW_MODE", "live").lower()
    if mode != "cache":
        try:
            rows = _live_search(api_query, k=k)
            if rows:
                return {"query": query, "api_query": api_query, "mode": "live", "results": rows[:k]}
            return {"query": query, "api_query": api_query, "mode": "live_empty", "results": []}
        except Exception as exc:
            cached = _cache_search(query, k=k)
            return {"query": query, "api_query": api_query, "mode": "cache_after_error", "error": str(exc), "results": cached}
    return {"query": query, "api_query": api_query, "mode": "cache", "results": _cache_search(query, k=k)}


def answer_stackoverflow(question: str, k: int = 3) -> dict:
    result = search_stackoverflow(question, k=k)
    rows = result["results"]
    evidence = [
        {
            "source": row.get("source", "stackoverflow"),
            "quote": row["summary"],
            "url": row["url"],
        }
        for row in rows
    ]
    answer = "По Stack Overflow / Stack Exchange релевантный практический совет: "
    answer += " ".join(row["summary"] for row in rows[:2]) if rows else "live API не вернул подходящих результатов."
    return {"answer": answer, "evidence": evidence, "raw": result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(search_stackoverflow(args.query, k=args.k), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
