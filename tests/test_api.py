import pytest
from pydantic import ValidationError

from app.schemas.request import (
    GenerationRequest,
    GenerationResponse,
)


# =========================================================
# GENERATION REQUEST TESTS
# =========================================================

def test_valid_generation_request():
    """
    Verify that a valid generation request
    is accepted by the request schema.
    """

    request = GenerationRequest(
        instruction="Solve the following problem.",
        input="What is 2 + 2?",
        max_new_tokens=256,
    )

    assert request.instruction == "Solve the following problem."
    assert request.input == "What is 2 + 2?"
    assert request.max_new_tokens == 256


def test_default_max_new_tokens():
    """
    Verify that max_new_tokens defaults to 512
    when the client does not provide a value.
    """

    request = GenerationRequest(
        instruction="Solve the problem.",
        input="What is 2 + 2?",
    )

    assert request.max_new_tokens == 512


def test_empty_instruction_rejected():
    """
    An empty instruction should be rejected
    before the request reaches the model.
    """

    with pytest.raises(ValidationError):
        GenerationRequest(
            instruction="",
            input="What is 2 + 2?",
        )


def test_empty_input_rejected():
    """
    An empty input should be rejected
    before the request reaches the model.
    """

    with pytest.raises(ValidationError):
        GenerationRequest(
            instruction="Solve the problem.",
            input="",
        )


def test_zero_max_new_tokens_rejected():
    """
    max_new_tokens must be at least 1.
    """

    with pytest.raises(ValidationError):
        GenerationRequest(
            instruction="Solve the problem.",
            input="What is 2 + 2?",
            max_new_tokens=0,
        )


def test_negative_max_new_tokens_rejected():
    """
    Negative generation limits should be rejected.
    """

    with pytest.raises(ValidationError):
        GenerationRequest(
            instruction="Solve the problem.",
            input="What is 2 + 2?",
            max_new_tokens=-10,
        )


def test_excessive_max_new_tokens_rejected():
    """
    max_new_tokens cannot exceed the configured
    schema limit of 2048.
    """

    with pytest.raises(ValidationError):
        GenerationRequest(
            instruction="Solve the problem.",
            input="What is 2 + 2?",
            max_new_tokens=2049,
        )


# =========================================================
# GENERATION RESPONSE TEST
# =========================================================

def test_generation_response_schema():
    """
    Verify that a valid model response containing
    token usage information is accepted.
    """

    response = GenerationResponse(
        response="Reasoning:\n2 + 2 = 4\n\nFinal Answer:\n4",
        input_tokens=49,
        output_tokens=16,
        total_tokens=65,
    )

    assert response.response != ""
    assert response.input_tokens == 49
    assert response.output_tokens == 16
    assert response.total_tokens == 65