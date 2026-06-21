# Двухуровневый Q&A ИИ-Ассистент по статьям про RAG

Проект для трека B: CLI-ассистент отвечает на вопросы про RAG по локальному корпусу Habr и при необходимости использует Stack Overflow fallback.

Одна команда запуска eval:

```bash
python eval.py
```

Что где лежит:

- `input/articles/` — 10 Habr-статей про RAG, LLM testing, DCD, OCC-RAG и оценку RAG.
- `input/gold.json` — 20 тестовых вопросов: `local_only`, `stackoverflow`, `mixed`, `out_of_scope`.
- `input/stackoverflow_cache.json` — аварийный/офлайн-кэш для Stack Overflow fallback.
- `rag.py` — локальный BM25 RAG по Habr-корпусу.
- `stackoverflow_tool.py` — Stack Overflow tool через кэш или Stack Exchange API.
- `agent.py` — PWC-паттерн: planner, parallel workers, critic.
- `schema.py` — Pydantic-схемы структурированного ответа и eval.
- `judge.py` — LLM-as-judge при `OPENAI_API_KEY` и детерминированный fallback без ключа.
- `eval.py` — прогон 20 вопросов, метрики правильности, пути, judge verdict и hallucination checks.
- `output/` — артефакты прогона: `eval_results.json`, `eval_table.csv`, `eval_table.md`, `trace.json`, `summary.json`, `hallucination_report.json`, `ghost_tests.json`.

Примеры:

```bash
python rag.py stats
python rag.py search "Что такое DCD?"
python stackoverflow_tool.py "hybrid search bm25 vector search rag"
python agent.py "Сравни DCD и hybrid search" --type mixed
python eval.py --ghost-tests
```

По умолчанию проект не требует API-ключей, но Stack Overflow tool ходит в live Stack Exchange API. Для полностью офлайн-прогона выставьте `STACKOVERFLOW_MODE=cache`. Для LLM-рафинирования структурированного ответа выставьте `OPENAI_API_KEY` и запустите `python eval.py --llm`. Для настоящего LLM-as-judge используйте `python eval.py --llm-judge`.
