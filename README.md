# AI Engineer Assessment

## Setup

```bash
pip install -e ".[dev]"
```

## Structure

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

```bash
pytest
```
