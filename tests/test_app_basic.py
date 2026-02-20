import json
import os

import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home_page_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Cosmic AI" in resp.data


@pytest.mark.parametrize(
    "path",
    [
        "/chat",
        "/converter",
        "/summarizer",
        "/image-generator",
        "/pricing",
        "/security",
        "/faq",
        "/privacy",
        "/terms",
    ],
)
def test_feature_pages_render(client, path):
    resp = client.get(path)
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

