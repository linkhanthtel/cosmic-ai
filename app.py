import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from chatbot import ChatBot
import db

templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = Path("uploads/tmp")
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_VIDEO_BYTES = 50 * 1024 * 1024

ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
}
ALLOWED_VIDEO_TYPES = {
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
}
ALLOWED_TYPES = {**ALLOWED_IMAGE_TYPES, **ALLOWED_VIDEO_TYPES}

_chatbot: ChatBot | None = None


def get_chatbot() -> ChatBot:
    global _chatbot
    if _chatbot is None:
        _chatbot = ChatBot()
    return _chatbot


def ensure_upload_dir() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def media_kind(content_type: str) -> str:
    if content_type in ALLOWED_IMAGE_TYPES:
        return "image"
    if content_type in ALLOWED_VIDEO_TYPES:
        return "video"
    raise HTTPException(status_code=400, detail="Unsupported file type")


def safe_filename(file_id: str) -> Path | None:
    if not re.fullmatch(r"[a-f0-9-]{36}", file_id):
        return None
    matches = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    if not matches:
        return None
    return matches[0]


def media_path_from_url(url: str) -> Path | None:
    match = re.match(r"^/media/([a-f0-9-]{36})$", url.strip())
    if not match:
        return None
    return safe_filename(match.group(1))


class Attachment(BaseModel):
    url: str
    media_type: str
    name: str = ""


class ChatRequest(BaseModel):
    message: str = ""
    conversation_id: str | None = None
    attachments: list[Attachment] = Field(default_factory=list)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_upload_dir()
    db.init_db()
    yield


app = FastAPI(title="Cosmic AI", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return RedirectResponse(url="/chat", status_code=302)


@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    return templates.TemplateResponse(request, "chat.html", {})


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request):
    return templates.TemplateResponse(request, "settings.html", {})


@app.post("/upload")
async def upload_media(file: UploadFile = File(...)):
    if not file.content_type or file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use jpg, png, gif, webp, mp4, webm, or mov.",
        )

    kind = media_kind(file.content_type)
    max_bytes = MAX_IMAGE_BYTES if kind == "image" else MAX_VIDEO_BYTES
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > max_bytes:
        limit_mb = max_bytes // (1024 * 1024)
        raise HTTPException(status_code=400, detail=f"File too large (max {limit_mb} MB)")

    ensure_upload_dir()
    file_id = str(uuid.uuid4())
    ext = ALLOWED_TYPES[file.content_type]
    path = UPLOAD_DIR / f"{file_id}{ext}"
    path.write_bytes(data)

    return {
        "id": file_id,
        "url": f"/media/{file_id}",
        "media_type": kind,
        "name": file.filename or f"upload{ext}",
        "content_type": file.content_type,
    }


@app.get("/media/{file_id}")
def get_media(file_id: str):
    path = safe_filename(file_id)
    if path is None:
        raise HTTPException(status_code=404, detail="File not found")

    suffix = path.suffix.lower()
    content_type = next(
        (mime for mime, ext in ALLOWED_TYPES.items() if ext == suffix),
        "application/octet-stream",
    )
    return FileResponse(path, media_type=content_type)


def conversation_id_or_400(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    if not re.fullmatch(r"[a-f0-9-]{36}", value):
        raise HTTPException(status_code=400, detail="Invalid conversation id")
    return value


def attachment_payload(attachments: list[Attachment]) -> list[dict]:
    return [
        {"url": item.url, "media_type": item.media_type, "name": item.name}
        for item in attachments
    ]


@app.get("/conversations")
def list_conversations():
    return {"conversations": db.list_conversations()}


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    conversation_id = conversation_id_or_400(conversation_id)
    conversation = db.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    conversation_id = conversation_id_or_400(conversation_id)
    if not db.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"ok": True}


@app.post("/chat")
def chat(body: ChatRequest):
    message = body.message.strip()
    attachments = body.attachments
    conversation_id = conversation_id_or_400(body.conversation_id)

    if not message and not attachments:
        raise HTTPException(status_code=400, detail="Add a message or attach a file")

    image_paths: list[Path] = []
    video_names: list[str] = []
    for item in attachments:
        if item.media_type == "image":
            path = media_path_from_url(item.url)
            if path is None:
                raise HTTPException(status_code=400, detail=f"Image not found: {item.name or item.url}")
            image_paths.append(path)
        elif item.media_type == "video":
            video_names.append(item.name or "video")

    try:
        bot = get_chatbot()
        if image_paths or video_names:
            response = bot.get_response_with_attachments(message, image_paths, video_names)
        else:
            response = bot.get_response(message)
    except Exception as e:
        db.add_log("chat_error", {"detail": str(e)})
        raise HTTPException(status_code=500, detail=str(e)) from e

    saved_attachments = attachment_payload(attachments)
    title = db.make_title(message, saved_attachments)

    if conversation_id:
        existing = db.get_conversation(conversation_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        if existing["title"] in ("New Chat", ""):
            db.set_conversation_title(conversation_id, title)
    else:
        created = db.create_conversation(title)
        conversation_id = created["id"]

    db.add_message(conversation_id, "user", message, saved_attachments)
    db.add_message(conversation_id, "assistant", response, [])
    conversation = db.get_conversation(conversation_id)

    return {
        "response": response,
        "timestamp": datetime.now().isoformat(),
        "conversation_id": conversation_id,
        "title": conversation["title"] if conversation else title,
    }


@app.post("/conversation/clear")
def clear_conversation():
    return {"message": get_chatbot().clear_conversation()}


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    reload = os.environ.get("DEBUG", "false").lower() == "true"
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=reload)
