"""Collect a Seminar 4 homework corpus from Habr articles about RAG.

The script builds 5-15 text documents:
- source: Habr search/tag pages for the query/tag "RAG";
- every saved document has 500-5000 words;
- the whole corpus has at least 30 000 characters;
- files are written as UTF-8 .txt into data/.

Usage:
    python collect_habr_corpus.py
    python collect_habr_corpus.py --tag RAG --target-docs 10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path
from typing import Iterable
from urllib.parse import quote, urljoin, urlparse
from xml.etree import ElementTree

import httpx


BASE_URL = "https://habr.com"
WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+(?:[-'][A-Za-zА-Яа-яЁё0-9]+)?")
TAG_RE = re.compile(r"<[^>]+>")
ARTICLE_LINK_RE = re.compile(r"https?://habr\.com/(?:ru|en)/(?:articles|companies/[^/]+/articles)/\d+/?")


@dataclass(frozen=True)
class Seed:
    url: str
    title: str = ""
    published: str = ""
    tags: tuple[str, ...] = ()


@dataclass
class Article:
    url: str
    title: str
    published: str
    tags: tuple[str, ...]
    text: str

    @property
    def words(self) -> int:
        return count_words(self.text)

    @property
    def chars(self) -> int:
        return len(self.text)


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def clean_spaces(text: str) -> str:
    text = unicodedata.normalize("NFKC", unescape(text))
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_tags(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript|svg|form|button).*?</\1>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>", "\n", html)
    html = re.sub(r"(?i)</(p|div|li|h2|h3|blockquote|pre)\s*>", "\n\n", html)
    return clean_spaces(TAG_RE.sub(" ", html))


def slugify(text: str, max_len: int = 56) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return (text[:max_len].strip("_") or "document")


def request_text(client: httpx.Client, url: str) -> str:
    response = client.get(url)
    response.raise_for_status()
    return response.text


def safe_print(message: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    print(message.encode(encoding, errors="replace").decode(encoding))


def parse_date(value: str) -> str:
    if not value:
        return ""
    try:
        return parsedate_to_datetime(value).isoformat()
    except Exception:
        return value.strip()


def parse_rss(xml_text: str) -> list[Seed]:
    seeds: list[Seed] = []
    root = ElementTree.fromstring(xml_text)
    for item in root.findall(".//item"):
        link = clean_spaces(item.findtext("link") or "")
        title = clean_spaces(item.findtext("title") or "")
        published = parse_date(item.findtext("pubDate") or "")
        categories = tuple(
            clean_spaces(category.text or "")
            for category in item.findall("category")
            if clean_spaces(category.text or "")
        )
        if "/articles/" in link:
            seeds.append(Seed(url=link, title=title, published=published, tags=categories))
    return seeds


def discovery_urls(tag: str, pages: int) -> list[str]:
    quoted = quote(tag)
    bracketed = quote(f"[{tag}]")
    urls = [
        f"{BASE_URL}/ru/rss/search/?q={quoted}&target_type=posts&order=date",
        f"{BASE_URL}/ru/rss/search/?q={bracketed}&target_type=posts&order=date",
        f"{BASE_URL}/ru/rss/search/?q={quoted}&target_type=posts&order=relevance",
        f"{BASE_URL}/ru/rss/hubs/artificial_intelligence/articles/all/",
        f"{BASE_URL}/ru/rss/hubs/machine_learning/articles/all/",
        f"{BASE_URL}/ru/rss/hubs/natural_language_processing/articles/all/",
    ]
    for page in range(2, pages + 1):
        urls.extend(
            [
                f"{BASE_URL}/ru/articles/page{page}/",
            ]
        )
    return urls


def normalize_article_url(url: str) -> str:
    url = url.split("#", 1)[0].split("?", 1)[0]
    parsed = urlparse(url)
    return parsed._replace(scheme="https", netloc="habr.com").geturl().rstrip("/") + "/"


def discover_from_html(html_text: str) -> list[Seed]:
    seeds: list[Seed] = []

    for href in re.findall(r"""href=["']([^"']+)["']""", html_text):
        url = urljoin(BASE_URL, href)
        if ARTICLE_LINK_RE.fullmatch(url.split("?", 1)[0].split("#", 1)[0]):
            seeds.append(Seed(url=normalize_article_url(url)))

    for match in ARTICLE_LINK_RE.finditer(html_text):
        seeds.append(Seed(url=normalize_article_url(match.group(0))))

    return seeds


def unique_seeds(seeds: Iterable[Seed]) -> list[Seed]:
    seen: set[str] = set()
    result: list[Seed] = []
    for seed in seeds:
        url = normalize_article_url(seed.url)
        if url in seen:
            continue
        seen.add(url)
        result.append(Seed(url=url, title=seed.title, published=seed.published, tags=seed.tags))
    return result


def collect_seeds(client: httpx.Client, tag: str, pages: int) -> list[Seed]:
    seeds: list[Seed] = []
    for url in discovery_urls(tag, pages):
        try:
            page_text = request_text(client, url)
        except Exception as exc:
            safe_print(f"Discovery skipped: {url} ({exc})")
            continue
        if "/rss/" in url:
            found = parse_rss(page_text)
        else:
            found = discover_from_html(page_text)
        seeds.extend(found)
        safe_print(f"Discovery ok: {url} ({len(found)} links)")
    return unique_seeds(seeds)


def extract_meta(html_text: str, prop: str) -> str:
    patterns = (
        rf'<meta[^>]+property=["\']{re.escape(prop)}["\'][^>]+content=["\']([^"\']+)["\']',
        rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']{re.escape(prop)}["\']',
        rf'<meta[^>]+name=["\']{re.escape(prop)}["\'][^>]+content=["\']([^"\']+)["\']',
    )
    for pattern in patterns:
        match = re.search(pattern, html_text, flags=re.I)
        if match:
            return clean_spaces(match.group(1))
    return ""


def extract_jsonld(html_text: str) -> tuple[str, str, tuple[str, ...]]:
    for raw in re.findall(
        r'(?is)<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html_text,
    ):
        try:
            parsed = json.loads(unescape(raw).strip())
        except json.JSONDecodeError:
            continue
        objects = parsed if isinstance(parsed, list) else [parsed]
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            title = clean_spaces(str(obj.get("headline") or obj.get("name") or ""))
            published = clean_spaces(str(obj.get("datePublished") or obj.get("dateModified") or ""))
            keywords = obj.get("keywords") or []
            if isinstance(keywords, str):
                tags = tuple(clean_spaces(t) for t in keywords.split(",") if clean_spaces(t))
            elif isinstance(keywords, list):
                tags = tuple(clean_spaces(str(t)) for t in keywords if clean_spaces(str(t)))
            else:
                tags = ()
            if title or published or tags:
                return title, published, tags
    return "", "", ()


def extract_title(html_text: str) -> str:
    title = extract_meta(html_text, "og:title")
    if title:
        return title.removesuffix(" / Хабр").strip()
    match = re.search(r"(?is)<h1[^>]*>(.*?)</h1>", html_text)
    if match:
        return strip_tags(match.group(1))
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", html_text)
    return strip_tags(match.group(1)).removesuffix(" / Хабр").strip() if match else ""


def extract_tags(html_text: str) -> tuple[str, ...]:
    tags: list[str] = []
    for match in re.findall(r'(?is)<a[^>]+href=["\'][^"\']*/tags/[^"\']+["\'][^>]*>(.*?)</a>', html_text):
        tag = strip_tags(match)
        if tag and tag not in tags:
            tags.append(tag)
    return tuple(tags)


def extract_article_body(html_text: str) -> str:
    patterns = (
        r'(?is)<div[^>]+id=["\']post-content-body["\'][^>]*>(.*?)</div>\s*</div>',
        r'(?is)<div[^>]+class=["\'][^"\']*tm-article-body[^"\']*["\'][^>]*>(.*?)</article>',
        r'(?is)<article[^>]*>(.*?)</article>',
    )
    for pattern in patterns:
        match = re.search(pattern, html_text)
        if match:
            text = strip_tags(match.group(1))
            if count_words(text) >= 200:
                return text

    paragraphs = [strip_tags(p) for p in re.findall(r"(?is)<p[^>]*>(.*?)</p>", html_text)]
    paragraphs = [p for p in paragraphs if count_words(p) >= 5]
    return clean_spaces("\n\n".join(paragraphs))


def fetch_article(client: httpx.Client, seed: Seed) -> Article | None:
    html_text = request_text(client, seed.url)
    jsonld_title, jsonld_published, jsonld_tags = extract_jsonld(html_text)
    title = jsonld_title or seed.title or extract_title(html_text)
    published = jsonld_published or seed.published or extract_meta(html_text, "article:published_time")
    tags = jsonld_tags or seed.tags or extract_tags(html_text)
    text = extract_article_body(html_text)
    if not text:
        return None
    return Article(
        url=seed.url,
        title=clean_spaces(title or seed.url.rstrip("/").rsplit("/", 1)[-1]),
        published=clean_spaces(published),
        tags=tuple(t for t in tags if t),
        text=clean_spaces(text),
    )


def is_relevant(article: Article, tag: str) -> bool:
    tag_lower = tag.lower()
    if any(t.lower() == tag_lower for t in article.tags):
        return True

    haystack = f"{article.title}\n{article.text}".lower()
    exact_tag = re.compile(rf"(?<![a-zа-яё0-9]){re.escape(tag_lower)}(?![a-zа-яё0-9])", re.I)
    return bool(exact_tag.search(haystack)) or "retrieval-augmented generation" in haystack


def document_text(article: Article) -> str:
    return clean_spaces(
        "\n".join(
            [
                f"# {article.title}",
                "",
                f"Источник: {article.url}",
                f"Дата публикации: {article.published or 'не указана'}",
                f"Теги: {', '.join(article.tags) if article.tags else 'не указаны'}",
                f"Слов: {article.words}",
                f"Символов: {article.chars}",
                "",
                article.text,
            ]
        )
    ) + "\n"


def save_corpus(articles: list[Article], output_dir: Path, keep_existing: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not keep_existing:
        for path in output_dir.glob("*.txt"):
            path.unlink()

    for index, article in enumerate(articles, start=1):
        filename = f"habr_{index:02d}_{slugify(article.title)}.txt"
        (output_dir / filename).write_text(document_text(article), encoding="utf-8")


def choose_documents(
    articles: list[Article],
    target_docs: int,
    min_words: int,
    max_words: int,
    min_total_chars: int,
) -> list[Article]:
    valid = [a for a in articles if min_words <= a.words <= max_words]
    valid.sort(key=lambda a: (a.published, a.chars), reverse=True)

    selected: list[Article] = []
    seen_urls: set[str] = set()
    for article in valid:
        if article.url in seen_urls:
            continue
        selected.append(article)
        seen_urls.add(article.url)
        if len(selected) >= target_docs and sum(a.chars for a in selected) >= min_total_chars:
            break
        if len(selected) >= 15:
            break
    return selected


def validate_corpus(
    articles: list[Article],
    min_docs: int,
    max_docs: int,
    min_words: int,
    max_words: int,
    min_total_chars: int,
) -> None:
    errors = []
    if not min_docs <= len(articles) <= max_docs:
        errors.append(f"documents: expected {min_docs}-{max_docs}, got {len(articles)}")
    for article in articles:
        if not min_words <= article.words <= max_words:
            errors.append(f"{article.title}: {article.words} words")
    total_chars = sum(article.chars for article in articles)
    if total_chars < min_total_chars:
        errors.append(f"total chars: expected at least {min_total_chars}, got {total_chars}")
    if errors:
        raise RuntimeError("Corpus requirements are not met:\n- " + "\n- ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="RAG")
    parser.add_argument("--pages", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--target-docs", type=int, default=10)
    parser.add_argument("--min-docs", type=int, default=5)
    parser.add_argument("--max-docs", type=int, default=15)
    parser.add_argument("--min-words", type=int, default=500)
    parser.add_argument("--max-words", type=int, default=5000)
    parser.add_argument("--min-total-chars", type=int, default=30000)
    parser.add_argument("--max-candidates", type=int, default=80)
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    if not 5 <= args.target_docs <= 15:
        parser.error("--target-docs must be between 5 and 15")

    headers = {
        "User-Agent": "Mozilla/5.0 corpus collector for educational RAG homework",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ru,en;q=0.8",
    }

    with httpx.Client(headers=headers, timeout=25.0, follow_redirects=True) as client:
        seeds = collect_seeds(client, tag=args.tag, pages=args.pages)[: args.max_candidates]
        safe_print(f"\nFound candidate URLs: {len(seeds)}")

        articles: list[Article] = []
        for index, seed in enumerate(seeds, start=1):
            try:
                article = fetch_article(client, seed)
            except Exception as exc:
                safe_print(f"[{index:03d}] skipped fetch error: {seed.url} ({exc})")
                continue
            if not article:
                safe_print(f"[{index:03d}] skipped empty: {seed.url}")
                continue
            if not is_relevant(article, args.tag):
                safe_print(f"[{index:03d}] skipped not relevant | {article.title[:90]}")
                continue
            tag_mark = "tag-ok" if any(t.lower() == args.tag.lower() for t in article.tags) else "tag?"
            articles.append(article)
            safe_print(f"[{index:03d}] {article.words:4d} words | {tag_mark} | {article.title[:90]}")
            time.sleep(args.delay)

    selected = choose_documents(
        articles=articles,
        target_docs=args.target_docs,
        min_words=args.min_words,
        max_words=args.max_words,
        min_total_chars=args.min_total_chars,
    )
    validate_corpus(
        selected,
        min_docs=args.min_docs,
        max_docs=args.max_docs,
        min_words=args.min_words,
        max_words=args.max_words,
        min_total_chars=args.min_total_chars,
    )
    save_corpus(selected, args.output_dir, keep_existing=args.keep_existing)

    total_words = sum(a.words for a in selected)
    total_chars = sum(a.chars for a in selected)
    safe_print("\nSaved corpus:")
    safe_print(f"  path: {args.output_dir}")
    safe_print(f"  documents: {len(selected)}")
    safe_print(f"  words: {total_words}")
    safe_print(f"  chars: {total_chars}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
