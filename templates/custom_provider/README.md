# Custom Provider Template

Extension template for implementing custom Atlas providers.

## Directory structure

```text
custom_provider/
├── README.md
├── requirements.txt
└── app/
    ├── __init__.py
    ├── embedding_provider.py    # Custom EmbeddingProvider implementation
    ├── reranker.py              # Custom Reranker implementation
    └── main.py                  # Demo usage
```

## Quick start

```bash
pip install -r requirements.txt
python app/main.py
```

## What it demonstrates

- Implementing a custom `EmbeddingProvider` subclass
- Implementing a custom `Reranker` subclass
- Using custom providers with `KnowledgeBase`
- Registering custom providers in the global registry

## Creating your own provider

1. Subclass `EmbeddingProvider` from `app.rag.embeddings.base`
2. Implement `name`, `embed()`, and `embed_batch()`
3. Register with `register_provider("my_provider", MyProvider)`
4. Instantiate and use in your pipeline
