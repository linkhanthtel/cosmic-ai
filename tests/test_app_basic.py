import json

import pytest
from fastapi.testclient import TestClient

from app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_root_redirects_to_chat(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 302
    assert "/chat" in resp.headers.get("location", "")


def test_chat_page_renders(client):
    resp = client.get("/chat")
    assert resp.status_code == 200


def test_chat_endpoint_basic(client):
    payload = {"message": "Where is Myanmar?"}
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    assert isinstance(data["response"], str)
