from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    """
    Input received by the model inference API.
    """

    instruction: str = Field(
        ...,
        min_length=1,
        description="Instruction describing the task.",
    )

    input: str = Field(
        ...,
        min_length=1,
        description="User input or problem to solve.",
    )

    max_new_tokens: int = Field(
        default=512,
        ge=1,
        le=2048,
        description="Maximum number of tokens to generate.",
    )


class GenerationResponse(BaseModel):
    """
    Output returned by the model inference API.

    Besides the generated response, this schema also
    exposes token usage information for monitoring
    and performance analysis.
    """

    # Generated text returned by the model.
    response: str

    # Number of tokens in the generated output.
    output_tokens: int

    # Number of input/prompt tokens.
    input_tokens: int

    # Total tokens processed.
    total_tokens: int