import os
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from chatbot import ChatBot

app = FastAPI(title="Cosmic AI")
templates = Jinja2Templates(directory="templates")

chatbot = ChatBot()

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("templates", exist_ok=True)


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

        response = chatbot.get_response(user_message)
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

        result = chatbot.train_model(body.training_data)
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

        if chatbot.add_training_data(question, answer):
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
        training_data = chatbot.get_training_data()
        return {"training_data": training_data, "count": len(training_data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/delete_data")
def delete_training_data(body: DeleteDataRequest):
    try:
        if body.index is None:
            raise HTTPException(status_code=400, detail="Index is required")

        if chatbot.delete_training_data(body.index):
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
        result = chatbot.retrain_model()
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
        summary = chatbot.get_conversation_summary()
        return {
            "summary": summary,
            "topics": chatbot.conversation_topics,
            "message_count": len(
                [entry for entry in chatbot.conversation_history if "user" in entry]
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/conversation/clear")
def clear_conversation():
    try:
        message = chatbot.clear_conversation()
        return {"message": message, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/conversation/history")
def get_conversation_history():
    try:
        return {
            "history": chatbot.conversation_history[-10:],
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/personality")
def get_personality():
    try:
        return chatbot.get_personality_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/personality/adjust")
def adjust_personality(body: PersonalityAdjustRequest):
    try:
        if not body.trait or body.value is None:
            raise HTTPException(status_code=400, detail="Trait and value are required")

        result = chatbot.adjust_personality(body.trait, body.value)
        return {
            "message": result,
            "personality": chatbot.get_personality_info(),
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
