# Cosmic AI Chat

AI chatbot with **LangChain RAG** over your Q&A training data, built with FastAPI.

## How it works

1. **Training data** — Q&A pairs in `data/training_data.json`
2. **Embeddings** — `sentence-transformers/all-MiniLM-L6-v2` (local, no API key)
3. **Vector store** — FAISS index in `models/faiss_index/`
4. **Answers**
   - **Default (no API key):** returns the best matching answer from retrieved documents
   - **With `OPENAI_API_KEY`:** LangChain RAG + OpenAI synthesizes an answer from retrieved context

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env   # optional: add OPENAI_API_KEY
uvicorn app:app --reload --port 8080
# or: python app.py
```

Open `http://localhost:8080` (redirects to `/chat`).

First run downloads the embedding model and builds the FAISS index (may take a minute).

## Retrain the knowledge index

After editing `data/training_data.json`:

```bash
python train_model.py
# or POST /retrain from the API
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | No | Enables LLM RAG mode |
| `OPENAI_MODEL` | No | Default `gpt-4o-mini` |
| `OPENAI_TEMPERATURE` | No | Default `0.3` |
| `PORT` | No | Default `8080` |

## Tests

```bash
pip install pytest
pytest
```

## Deploy (Fly.io)

```bash
fly deploy
```

Use at least **1GB RAM** (embeddings + FAISS). Set secrets: `fly secrets set OPENAI_API_KEY=sk-...`

## Project structure

```
cosmic-ai/
├── app.py
├── chatbot.py          # LangChain + FAISS RAG
├── train_model.py
├── data/training_data.json
├── models/faiss_index/
├── templates/
├── tests/
├── Dockerfile
└── fly.toml
```
