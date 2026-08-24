from fastapi.testclient import TestClient

from app.main import app
from app.schemas.request import GenerationResponse


class FakeGenerationService:
    """
    Fake generation service used for API tests.

    This avoids loading the real Llama model.
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


# =========================================================
# HEALTH ENDPOINT
# =========================================================

def test_health_endpoint():
    """
    Verify that /health returns a healthy response.
    """

    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


# =========================================================
# READY ENDPOINT
# =========================================================

def test_ready_endpoint():
    """
    Verify that /ready reports the application as ready
    when the model is loaded.
    """

    client = TestClient(app)

    response = client.get("/ready")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ready"
    assert data["model_loaded"] is True


# =========================================================
# GENERATE ENDPOINT
# =========================================================

def test_generate_endpoint(monkeypatch):
    """
    Verify that /generate accepts a valid request and
    returns the expected response structure.
    """

    fake_service = FakeGenerationService()

    # Replace the real generation service with the fake one.
    app.state.generation_service = fake_service

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


# =========================================================
# INVALID GENERATE REQUEST
# =========================================================

def test_generate_empty_instruction():
    """
    Verify that /generate rejects an empty instruction.
    """

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