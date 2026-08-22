# Import the existing inference function.
# This function contains the actual Llama text-generation logic.
from app.models.inference import generate_response


# Import the request and response schemas.
# GenerationRequest  -> defines what data comes INTO the service.
# GenerationResponse -> defines what data comes OUT of the service.
from app.schemas.request import GenerationRequest, GenerationResponse


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
        Generate a model response for one request.

        Input:
            GenerationRequest
            - instruction
            - input
            - max_new_tokens

        Output:
            GenerationResponse
            - response

        This method connects the API request format
        with the existing model inference function.
        """

        # Call the existing inference function.
        #
        # We are NOT writing the generation logic again here.
        # The actual tokenization and model.generate() logic
        # already lives inside app/models/inference.py.
        response = generate_response(
            instruction=request.instruction,
            user_input=request.input,
            tokenizer=self.tokenizer,
            model=self.model,
            max_new_tokens=request.max_new_tokens,
        )

        # Convert the raw model output into our defined
        # GenerationResponse schema.
        #
        # This keeps the output format consistent for the API.
        return GenerationResponse(
            response=response
        )