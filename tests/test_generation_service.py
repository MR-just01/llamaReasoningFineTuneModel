import pytest

from app.services.generation_service import GenerationService
from app.schemas.request import GenerationRequest
from app.core.exceptions import (
    ModelNotReadyError,
    GenerationError,
)


# =========================================================
# SUCCESSFUL GENERATION
# =========================================================

def test_generation_service_success(monkeypatch):
    """
    Verify that the generation service successfully calls
    the inference function and returns a GenerationResponse.
    """

    # Fake inference function.
    def fake_generate_response(
        instruction,
        user_input,
        tokenizer,
        model,
        max_new_tokens,
    ):
        return "Reasoning:\n2 + 2 = 4\n\nFinal Answer:\n4"

    # Replace the real inference function with our fake one.
    monkeypatch.setattr(
        "app.services.generation_service.generate_response",
        fake_generate_response,
    )

    service = GenerationService(
        tokenizer="fake_tokenizer",
        model="fake_model",
    )

    request = GenerationRequest(
        instruction="Solve the problem.",
        input="What is 2 + 2?",
        max_new_tokens=256,
    )

    result = service.generate(request)

    assert result.response == (
        "Reasoning:\n2 + 2 = 4\n\nFinal Answer:\n4"
    )


# =========================================================
# MODEL NOT READY
# =========================================================

def test_generation_service_model_not_ready():
    """
    Verify that the service rejects generation when the
    model or tokenizer has not been loaded.
    """

    service = GenerationService(
        tokenizer=None,
        model=None,
    )

    request = GenerationRequest(
        instruction="Solve the problem.",
        input="What is 2 + 2?",
    )

    with pytest.raises(ModelNotReadyError):
        service.generate(request)


# =========================================================
# INFERENCE FAILURE
# =========================================================

def test_generation_service_inference_failure(monkeypatch):
    """
    Verify that low-level inference errors are converted
    into our application-specific GenerationError.
    """

    def fake_generate_response(
        instruction,
        user_input,
        tokenizer,
        model,
        max_new_tokens,
    ):
        raise RuntimeError("Fake inference failure")

    monkeypatch.setattr(
        "app.services.generation_service.generate_response",
        fake_generate_response,
    )

    service = GenerationService(
        tokenizer="fake_tokenizer",
        model="fake_model",
    )

    request = GenerationRequest(
        instruction="Solve the problem.",
        input="What is 2 + 2?",
    )

    with pytest.raises(GenerationError):
        service.generate(request)