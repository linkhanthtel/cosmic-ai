# Cosmic AI Chat

AI chatbot with **LangChain RAG** over your Q&A training data, built with FastAPI.

## How it works

1. **Training data** — Q&A pairs in `data/training_data.json`
2. **Embeddings** — `sentence-transformers/all-MiniLM-L6-v2` (local, no API key)
3. **Vector store** — FAISS index in `models/faiss_index/`
4. **Answers** — `chatbot.py` auto-detects the best available mode:
   - **With `HF_TOKEN` (free, recommended):** LangChain RAG + a free Hugging Face LLM (`ChatHuggingFace`) synthesizes an answer from retrieved context
   - **With `OPENAI_API_KEY`:** LangChain RAG + OpenAI synthesizes the answer
   - **Neither set:** retrieval-only — returns the best matching stored answer

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env   # optional: OPENAI_API_KEY, RETRIEVAL_MAX_DISTANCE
uvicorn app:app --reload --port 8080
# or: python app.py
```

Open `http://localhost:8080` (redirects to `/chat`).

- First startup loads embeddings + FAISS (~10–30s). Watch the terminal for `Cosmic AI ready`.
- Check status: `curl http://localhost:8080/health`
- Learning script (Ollama agents, separate): `python ollama-learn.py`

## Retrain the knowledge index

After editing `data/training_data.json`:

```bash
python train_model.py
# or POST /retrain from the API
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | No | Free Hugging Face token; enables the free HF LLM RAG mode (and speeds up embedding downloads) |
| `HF_MODEL` | No | Default `meta-llama/Llama-3.1-8B-Instruct` (also try `Qwen/Qwen2.5-7B-Instruct`) |
| `HF_TEMPERATURE` | No | Default `0.3` |
| `HF_MAX_NEW_TOKENS` | No | Default `512` |
| `OPENAI_API_KEY` | No | Enables OpenAI RAG mode (used only if `HF_TOKEN` is not set) |
| `OPENAI_MODEL` | No | Default `gpt-4o-mini` |
| `OPENAI_TEMPERATURE` | No | Default `0.3` |
| `PORT` | No | Default `8080` |

## Tests

```bash
pip install pytest
pytest
```

## Deploy

### Memory: pick an embeddings backend

Running local embeddings loads PyTorch (~500 MB+ RAM). On small hosts (e.g. Render
free/starter 512 MB) this causes slow startups and out-of-memory crashes — which
show up in the browser as **"Knowledge base is still loading. Please try again in
a moment."** (the background loader never finishes).

Fix: set `EMBEDDINGS_BACKEND=hf_api` in production so embeddings are computed on
Hugging Face servers instead of locally. This keeps memory low enough for 512 MB
and makes startup fast. It requires `HF_TOKEN`.

### Render

`render.yaml` already sets `EMBEDDINGS_BACKEND=hf_api` and `HF_MODEL`. In the Render
dashboard set the secret `HF_TOKEN` (type: Read token from Hugging Face). Redeploy.
Verify with `curl https://<your-app>.onrender.com/health` — `status` should become
`ok` (or `error` with a message if something failed).

### Fly.io

```bash
fly deploy
```

Either use `EMBEDDINGS_BACKEND=hf_api` (recommended, low memory) or give the machine
at least **1 GB RAM** for local embeddings. Set secrets:
`fly secrets set HF_TOKEN=hf_... EMBEDDINGS_BACKEND=hf_api`

### Troubleshooting

- **"Knowledge base is still loading" forever** → startup is too slow or OOM-crashed.
  Set `EMBEDDINGS_BACKEND=hf_api` (+ `HF_TOKEN`), or use ≥1 GB RAM. Check `/health`
  for an `error` field, and check the deploy logs.
- **"model_not_supported" from Hugging Face** → change `HF_MODEL` (e.g.
  `meta-llama/Llama-3.1-8B-Instruct` or `Qwen/Qwen2.5-7B-Instruct`).

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

flowchart LR
  User["User question"] --> Embed["Embed question"]
  Embed --> FAISS["FAISS search"]
  KB["training_data.json"] --> Index["Vector index"]
  Index --> FAISS
  FAISS --> Context["Top matching Q&A"]
  Context --> Mode{OPENAI_API_KEY?}
  Mode -->|No| Return["Return stored answer"]
  Mode -->|Yes| LLM["OpenAI + context"]
  LLM --> Answer["Final reply"]
  Return --> Answer

