# Cosmic AI (Under Development)

Chat UI + LangChain practice app using a Hugging Face model**.

## What you have

```
cosmic-ai/
├── app.py              # FastAPI server + chat UI routes
├── chatbot.py          # LangChain chain (edit this to practice)
├── templates/          # Chat UI
├── Dockerfile
└── fly.toml
```

The LangChain chain lives in `chatbot.py`:

```python
self.chain = self.prompt | self.llm | StrOutputParser()
```

Try adding memory, RAG, tools, or streaming there.

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env   # add your HF_TOKEN
python app.py
```

Open http://localhost:8080

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes | Free Hugging Face token from https://huggingface.co/settings/tokens (type: Read) |
| `HF_MODEL` | No | Default `meta-llama/Llama-3.1-8B-Instruct` (also try `Qwen/Qwen2.5-7B-Instruct`) |
| `HF_TEMPERATURE` | No | Default `0.3` |
| `HF_MAX_NEW_TOKENS` | No | Default `512` |
| `PORT` | No | Default `8080` |

## Deploy (Fly.io)

```bash
fly secrets set HF_TOKEN=hf_your-token-here
fly deploy
```

Verify:

```bash
curl https://cosmic-ai.fly.dev/health
```
