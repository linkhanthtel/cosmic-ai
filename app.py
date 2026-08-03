import os
import re
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from chatbot import ChatBot

templates = Jinja2Templates(directory="templates")
app = FastAPI(title="Cosmic AI")

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
    attachments: list[Attachment] = Field(default_factory=list)


@app.on_event("startup")
def on_startup():
    ensure_upload_dir()


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


@app.post("/chat")
def chat(body: ChatRequest):
    message = body.message.strip()
    attachments = body.attachments

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
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"response": response, "timestamp": datetime.now().isoformat()}


@app.post("/conversation/clear")
def clear_conversation():
    return {"message": get_chatbot().clear_conversation()}


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    reload = os.environ.get("DEBUG", "false").lower() == "true"
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=reload)
