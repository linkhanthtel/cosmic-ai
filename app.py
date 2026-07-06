import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from chatbot import ChatBot

chatbot: ChatBot | None = None
_chatbot_lock = threading.Lock()
_chatbot_loading = False
_chatbot_ready = False
templates = Jinja2Templates(directory="templates")

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("templates", exist_ok=True)


def get_chatbot() -> ChatBot:
    global chatbot
    if chatbot is None:
        with _chatbot_lock:
            if chatbot is None:
                chatbot = ChatBot()
    return chatbot


def _load_chatbot_background() -> None:
    global _chatbot_loading, _chatbot_ready
    if _chatbot_ready or _chatbot_loading:
        return
    _chatbot_loading = True
    try:
        print("Loading Cosmic AI knowledge base (embeddings + FAISS)...")
        bot = get_chatbot()
        samples = len(bot.training_data)
        ready = bot.retriever is not None
        mode = bot.get_personality_info().get("langchain_mode", "unknown")
        print(
            f"Cosmic AI ready: {samples} Q&A pairs, "
            f"index={'ok' if ready else 'missing'}, mode={mode}"
        )
        _chatbot_ready = True
    except Exception as exc:
        print(f"Cosmic AI failed to load knowledge base: {exc}")
    finally:
        _chatbot_loading = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load embeddings in the background so the server binds PORT immediately
    # (Render/Fly health checks require an open port within ~90s).
    threading.Thread(target=_load_chatbot_background, daemon=True).start()
    yield


app = FastAPI(title="Cosmic AI", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str = ""


class TrainRequest(BaseModel):
    training_data: list[dict[str, Any]] = Field(default_factory=list)


class AddDataRequest(BaseModel):
    question: str = ""
    answer: str = ""


class DeleteDataRequest(BaseModel):
    index: Optional[int] = None


class PersonalityAdjustRequest(BaseModel):
    trait: Optional[str] = None
    value: Optional[float] = None


@app.get("/health")
def health():
    if not _chatbot_ready and chatbot is None:
        return {
            "status": "starting",
            "training_samples": 0,
            "index_ready": False,
            "mode": "loading",
        }
    bot = get_chatbot()
    info = bot.get_personality_info()
    return {
        "status": "ok" if _chatbot_ready or bot.retriever is not None else "starting",
        "training_samples": len(bot.training_data),
        "index_ready": bot.retriever is not None,
        "mode": info.get("langchain_mode"),
    }


@app.get("/")
def index():
    return RedirectResponse(url="/chat", status_code=302)


@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    return templates.TemplateResponse(request, "chat.html", {})


@app.post("/chat")
def chat(body: ChatRequest):
    try:
        user_message = body.message.strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="No message provided")

        if not _chatbot_ready and chatbot is None:
            raise HTTPException(
                status_code=503,
                detail="Knowledge base is still loading. Please try again in a moment.",
            )

        response = get_chatbot().get_response(user_message)
        return {
            "response": response,
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/train")
def train(body: TrainRequest):
    try:
        if not body.training_data:
            raise HTTPException(status_code=400, detail="No training data provided")

        result = get_chatbot().train_model(body.training_data)
        return {
            "message": "Training completed successfully",
            "trained_samples": result["trained_samples"],
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/add_data")
def add_training_data(body: AddDataRequest):
    try:
        question = body.question.strip()
        answer = body.answer.strip()
        if not question or not answer:
            raise HTTPException(status_code=400, detail="Both question and answer are required")

        if get_chatbot().add_training_data(question, answer):
            return {
                "message": "Training data added successfully",
                "timestamp": datetime.now().isoformat(),
            }
        raise HTTPException(status_code=500, detail="Failed to add training data")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/get_training_data")
def get_training_data():
    try:
        training_data = get_chatbot().get_training_data()
        return {"training_data": training_data, "count": len(training_data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/delete_data")
def delete_training_data(body: DeleteDataRequest):
    try:
        if body.index is None:
            raise HTTPException(status_code=400, detail="Index is required")

        if get_chatbot().delete_training_data(body.index):
            return {
                "message": "Training data deleted successfully",
                "timestamp": datetime.now().isoformat(),
            }
        raise HTTPException(status_code=500, detail="Failed to delete training data")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/retrain")
def retrain():
    try:
        result = get_chatbot().retrain_model()
        return {
            "message": "Model retrained successfully",
            "trained_samples": result["trained_samples"],
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/conversation/summary")
def get_conversation_summary():
    try:
        bot = get_chatbot()
        summary = bot.get_conversation_summary()
        return {
            "summary": summary,
            "topics": bot.conversation_topics,
            "message_count": len(
                [entry for entry in bot.conversation_history if "user" in entry]
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/conversation/clear")
def clear_conversation():
    try:
        message = get_chatbot().clear_conversation()
        return {"message": message, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/conversation/history")
def get_conversation_history():
    try:
        bot = get_chatbot()
        return {
            "history": bot.conversation_history[-10:],
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/personality")
def get_personality():
    try:
        return get_chatbot().get_personality_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/personality/adjust")
def adjust_personality(body: PersonalityAdjustRequest):
    try:
        if not body.trait or body.value is None:
            raise HTTPException(status_code=400, detail="Trait and value are required")

        bot = get_chatbot()
        result = bot.adjust_personality(body.trait, body.value)
        return {
            "message": result,
            "personality": bot.get_personality_info(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    reload = os.environ.get("DEBUG", "false").lower() == "true"
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=reload)
