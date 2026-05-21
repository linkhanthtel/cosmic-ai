import json

import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_root_redirects_to_chat(client):
    resp = client.get("/")
    assert resp.status_code == 302
    assert "/chat" in resp.headers.get("Location", "")


def test_chat_page_renders(client):
    resp = client.get("/chat")
    assert resp.status_code == 200


def test_chat_endpoint_basic(client):
    payload = {"message": "Where is Myanmar?"}
    resp = client.post(
        "/chat",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert "response" in data
    assert isinstance(data["response"], str)
