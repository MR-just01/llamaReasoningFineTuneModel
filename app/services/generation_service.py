
import logging
import time     
# ModelNotReadyError -> model/tokenizer is unavailable.
# GenerationError    -> inference failed.
from app.core.exceptions import (
    ModelNotReadyError,
    GenerationError,
)


# Import the existing inference function.
# This function contains the actual Llama text-generation logic.
from app.models.inference import generate_response


# Import the request and response schemas.
#
# GenerationRequest  -> defines what data comes INTO the service.
# GenerationResponse -> defines what data goes OUT of the service.
from app.schemas.request import (
    GenerationRequest,
    GenerationResponse,
)

logger = logging.getLogger(__name__)


class GenerationService:
    """
    Service layer for model generation.

    Purpose:
    - Receive a validated generation request.
    - Pass the request to the model inference layer.
    - Return the generated response in the required format.

    The service layer acts as a bridge between:

        API layer
            ↓
        GenerationService
            ↓
        Inference layer
            ↓
        Llama model
    """

    def __init__(self, tokenizer, model):
        """
        Store the already-loaded tokenizer and model.

        IMPORTANT:
        We do NOT load the model here.

        The model should be loaded only once when the application
        starts and then reused for multiple requests.

        This prevents expensive model loading for every request.
        """

        # Tokenizer converts text into tokens that the model understands.
        self.tokenizer = tokenizer

        # Model is the fine-tuned Llama + LoRA model used for generation.
        self.model = model

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        """
        Generate a response using the fine-tuned model.
        """
        start_time = time.perf_counter()
        # ---------------------------------------------------------
        # CHECK MODEL AVAILABILITY
        # ---------------------------------------------------------
        #
        # If the model or tokenizer is missing, generation
        # cannot happen.
        #
        if self.model is None or self.tokenizer is None:
            raise ModelNotReadyError(
                "Model is not ready."
            )

        try:
            logger.info("Starting model generation.")
            # -----------------------------------------------------
            # RUN MODEL INFERENCE
            # -----------------------------------------------------
            #
            # Call the actual inference logic from inference.py.
            #
            response = generate_response(
                instruction=request.instruction,
                user_input=request.input,
                tokenizer=self.tokenizer,
                model=self.model,
                max_new_tokens=request.max_new_tokens,
            )
            latency = time.perf_counter() - start_time

            logger.info(
                "Model generation completed in %.3f seconds.",
                latency,
            )

        except Exception as exc:

            # -----------------------------------------------------
            # HANDLE INFERENCE FAILURE
            # -----------------------------------------------------
            #
            # Convert the low-level Python/model error into
            # an application-specific GenerationError.
            #
            # The original exception is preserved internally
            # using "from exc".
            latency = time.perf_counter() - start_time

            logger.exception(
        "Model generation failed after %.3f seconds.",
        latency,
    )
            raise GenerationError(
                "Model generation failed."
            ) from exc

        # ---------------------------------------------------------
        # RETURN STANDARD API RESPONSE
        # ---------------------------------------------------------
        #
        # Convert the raw model output into our defined
        # GenerationResponse schema.
        #
        return GenerationResponse(
            response=response
        )