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

    def fake_generate_response(
        instruction,
        user_input,
        tokenizer,
        model,
        max_new_tokens,
    ):
        return {
            "response": (
                "Reasoning:\n"
                "2 + 2 = 4\n\n"
                "Final Answer:\n"
                "4"
            ),
            "input_tokens": 49,
            "output_tokens": 16,
            "total_tokens": 65,
        }

    # Replace the real model inference with our fake function.
    monkeypatch.setattr(
        "app.services.generation_service.generate_response",
        fake_generate_response,
    )

    # Fake model/tokenizer.
    # The actual Llama model is NOT loaded.
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
        "Reasoning:\n"
        "2 + 2 = 4\n\n"
        "Final Answer:\n"
        "4"
    )

    assert result.input_tokens == 49
    assert result.output_tokens == 16
    assert result.total_tokens == 65


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