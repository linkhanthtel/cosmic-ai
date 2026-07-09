import os
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from chatbot import ChatBot

templates = Jinja2Templates(directory="templates")
app = FastAPI(title="Cosmic AI")

_chatbot: ChatBot | None = None


def get_chatbot() -> ChatBot:
    global _chatbot
    if _chatbot is None:
        _chatbot = ChatBot()
    return _chatbot


class ChatRequest(BaseModel):
    message: str = ""


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return RedirectResponse(url="/chat", status_code=302)


@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    return templates.TemplateResponse(request, "chat.html", {})


@app.post("/chat")
def chat(body: ChatRequest):
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="No message provided")
    try:
        response = get_chatbot().get_response(message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"response": response, "timestamp": datetime.now().isoformat()}


@app.post("/conversation/clear")
def clear_conversation():
    return {"message": get_chatbot().clear_conversation()}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    reload = os.environ.get("DEBUG", "false").lower() == "true"
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=reload)
