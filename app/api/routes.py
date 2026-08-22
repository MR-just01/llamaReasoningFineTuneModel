from fastapi import APIRouter, Request

from app.schemas.request import (
    GenerationRequest,
    GenerationResponse,
)


# Create the API router.
#
# The router contains the HTTP endpoints of our application.
router = APIRouter()


@router.post(
    "/generate",
    response_model=GenerationResponse,
)
def generate(
    request: GenerationRequest,
    http_request: Request,
):
    """
    Generate a response using the fine-tuned Llama model.

    Request:
        instruction
        input
        max_new_tokens

    Response:
        response
    """

    # Get the GenerationService that was created
    # during application startup.
    service = http_request.app.state.generation_service

    # Pass the validated request to the service layer.
    return service.generate(request)