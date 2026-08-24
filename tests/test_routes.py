

import app.main as main_module

from fastapi.testclient import TestClient

from app.main import app
from app.schemas.request import GenerationResponse


class FakeGenerationService:
    """
    Fake generation service used for API tests.

    The real Llama model is not loaded.
    """

    def generate(self, request):
        return GenerationResponse(
            response=(
                "Reasoning:\n"
                "2 + 2 = 4\n\n"
                "Final Answer:\n"
                "4"
            ),
            input_tokens=49,
            output_tokens=16,
            total_tokens=65,
        )


def test_health_endpoint(monkeypatch):

    monkeypatch.setattr(
        main_module,
        "model",
        "fake_model",
    )

    monkeypatch.setattr(
        main_module,
        "tokenizer",
        "fake_tokenizer",
    )

    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_ready_endpoint(monkeypatch):

    monkeypatch.setattr(
        main_module,
        "model",
        "fake_model",
    )

    monkeypatch.setattr(
        main_module,
        "tokenizer",
        "fake_tokenizer",
    )

    fake_service = FakeGenerationService()

    monkeypatch.setattr(
        app.state,
        "generation_service",
        fake_service,
        raising=False,
    )

    client = TestClient(app)

    response = client.get("/ready")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ready"
    assert data["model_loaded"] is True


def test_ready_endpoint_when_model_not_loaded(monkeypatch):

    monkeypatch.setattr(
        main_module,
        "model",
        None,
    )

    monkeypatch.setattr(
        main_module,
        "tokenizer",
        None,
    )

    monkeypatch.setattr(
        app.state,
        "generation_service",
        None,
        raising=False,
    )

    client = TestClient(app)

    response = client.get("/ready")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "not_ready"
    assert data["model_loaded"] is False


def test_generate_endpoint(monkeypatch):

    fake_service = FakeGenerationService()

    monkeypatch.setattr(
        app.state,
        "generation_service",
        fake_service,
        raising=False,
    )

    client = TestClient(app)

    response = client.post(
        "/generate",
        json={
            "instruction": "Solve the problem.",
            "input": "What is 2 + 2?",
            "max_new_tokens": 256,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["response"] == (
        "Reasoning:\n"
        "2 + 2 = 4\n\n"
        "Final Answer:\n"
        "4"
    )

    assert data["input_tokens"] == 49
    assert data["output_tokens"] == 16
    assert data["total_tokens"] == 65


def test_generate_empty_instruction():

    client = TestClient(app)

    response = client.post(
        "/generate",
        json={
            "instruction": "",
            "input": "What is 2 + 2?",
            "max_new_tokens": 256,
        },
    )

    assert response.status_code == 422