# RAG Application Template

Recommended production layout for an Atlas-based RAG application.

## Directory structure

```
rag_app/
├── README.md
├── requirements.txt
├── config.json
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── loader.py          # Custom document loader
│   └── search.py          # Search orchestration
└── data/                  # Document storage
    └── .gitkeep
```

## Quick start

```bash
pip install -r requirements.txt
python app/main.py
```

## Customization

1. Edit `config.json` to set environment and logging preferences.
2. Customize `app/loader.py` to load your documents.
3. Run with `python app/main.py`.

## What it demonstrates

- Configuration from JSON file via `ConfigLoader`
- `PipelineBuilder` for clean pipeline construction
- Separate loader module
- Structured logging with `AtlasLogger`
- Search orchestration with metadata
