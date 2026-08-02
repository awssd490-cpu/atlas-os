# Minimal Atlas Project

Smallest runnable Atlas project demonstrating a basic retrieval flow.

## Directory structure

```text
minimal/
├── README.md
├── requirements.txt
└── app/
    └── main.py
```

## Quick start

```bash
pip install -r requirements.txt
python app/main.py
```

## What it demonstrates

- Creating a `KnowledgeBase` with sample documents
- Running keyword retrieval with `KnowledgeRetriever`
- Building a formatted context with `KnowledgeContextBuilder`

## Expected output

```text
Registered 2 document(s)
Keyword search for 'capital of France': found 2 chunk(s)
Context: Paris is the capital of France.
```
