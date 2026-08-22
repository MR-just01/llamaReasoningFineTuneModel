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
    """

    response: str