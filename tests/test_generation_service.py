def test_generation_service_success(monkeypatch):
    """
    Verify that the generation service successfully calls
    the inference function and returns a GenerationResponse.
    """

    # Fake inference function.
    #
    # IMPORTANT:
    # This must return the same structure as the real
    # generate_response() function.
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

    # Replace the real inference function with our fake one.
    monkeypatch.setattr(
        "app.services.generation_service.generate_response",
        fake_generate_response,
    )

    # Create the service using fake model components.
    #
    # We don't load Llama during this unit test.
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

    # Verify generated response.
    assert result.response == (
        "Reasoning:\n"
        "2 + 2 = 4\n\n"
        "Final Answer:\n"
        "4"
    )

    # Verify token information is passed through correctly.
    assert result.input_tokens == 49
    assert result.output_tokens == 16
    assert result.total_tokens == 65