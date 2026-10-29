# AI Engineer Assessment

A compact Python project exploring the design and evaluation of LLM-powered question-answering systems. It pairs a direct OpenAI API baseline with a retrieval-augmented generation (RAG) pipeline, alongside tests that invite closer examination of correctness, reliability, and test quality.

## What This Project Covers

- Direct question answering with an LLM
- Document chunking and embedding-based retrieval
- Optional relevance reranking
- Pydantic models for pipeline inputs and results
- Unit testing with mocked model calls
- Code review topics including validation, async behavior, ranking, and API error handling

## Repository Layout

```text
src/
  qa.py          # Direct LLM question answering
  qa_rag.py      # RAG example using a small in-memory document set
rag/
  pipeline.py    # RAG pipeline implementation
  pipeline_pr.py # Pipeline version exercised by the tests
tests/
  test_qa.py
  test_pipeline.py
```

## Setup

Requires Python 3.11 or newer and an OpenAI API key.

```bash
python -m venv .venv
```

Activate the virtual environment, then install the project and development dependencies:

```bash
pip install -e ".[dev]"
```

Set `OPENAI_API_KEY` in your environment before running the examples. Some tests mock model calls, but importing the RAG example currently makes an embeddings API call, so running the full test suite may also require a valid key.

<<<<<<< Updated upstream
```
similar_cases/
  finder.py    # embedding-based search for comparable insurance cases
  models.py    # input and result contracts
tests/
  test_similar_case_finder.py
```

## Similar Case Finder

The Similar Case Finder indexes short summaries of fictional insurance cases.
Given a natural-language query, it returns the most similar cases using cosine
similarity over embeddings. Embedding providers are injected so tests can run
without external API calls.

## Run tests
=======
## Run Tests
>>>>>>> Stashed changes

```bash
pytest
```

## Notes

This repository is intended as an assessment and learning project, not a production-ready RAG service. The pipeline and tests provide concrete material for reviewing design choices and identifying areas for improvement.
