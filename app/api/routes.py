from fastapi import APIRouter, HTTPException, Request

from app.core.exceptions import (
    ModelNotReadyError,
    GenerationError,
)
from app.schemas.request import (
    GenerationRequest,
    GenerationResponse,
)


# Create the API router.
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

    The API route is responsible for:
    - Receiving the HTTP request.
    - Validating the request through Pydantic.
    - Calling the generation service.
    - Returning the API response.
    - Converting known application errors into HTTP errors.
    """

    # Get the service that was created during
    # application startup.
    service = http_request.app.state.generation_service

    try:
        # Send the validated request to the service layer.
        return service.generate(request)

    except ModelNotReadyError as exc:
        # Return a controlled HTTP 500 response instead
        # of exposing the internal Python traceback.
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
    except GenerationError as exc:

        # 500 means the model failed during generation.
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc