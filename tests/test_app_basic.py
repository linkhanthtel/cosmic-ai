import json
import time

import pytest
from fastapi.testclient import TestClient

from app import _chatbot_ready, app, get_chatbot


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        deadline = time.time() + 120
        while not _chatbot_ready and time.time() < deadline:
            time.sleep(0.2)
        if not _chatbot_ready:
            get_chatbot()
        yield test_client


def test_root_redirects_to_chat(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 302
    assert "/chat" in resp.headers.get("location", "")


def test_chat_page_renders(client):
    resp = client.get("/chat")
    assert resp.status_code == 200


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["index_ready"] is True
    assert data["training_samples"] > 0


def test_chat_endpoint_basic(client):
    payload = {"message": "Where is Myanmar?"}
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    assert isinstance(data["response"], str)
