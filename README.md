# Cosmic AI Chat

Keyword-based AI chatbot with training data management, built with Flask and scikit-learn.

## Installation

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:8080` (redirects to `/chat`).

## Usage

- Open **AI Chat** and send messages.
- Training data lives in `data/training_data.json`.
- Use `train_model.py` to train or update the model locally.

## Tests

```bash
pip install pytest
pytest
```

## Deploy (Fly.io)

```bash
fly deploy
```

Uses `Dockerfile` and `fly.toml`. The image only includes chat dependencies (Flask, scikit-learn, numpy, joblib).

## Project structure

```
cosmic-ai/
├── app.py              # Flask app (chat routes only)
├── chatbot.py          # Chatbot logic
├── train_model.py      # Offline model training
├── requirements.txt
├── data/training_data.json
├── models/             # Saved model files (optional)
├── templates/
│   ├── base.html
│   └── chat.html
├── tests/
├── Dockerfile
└── fly.toml
```
