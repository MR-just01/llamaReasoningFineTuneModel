# Load the fine-tuned Llama model and tokenizer.
# The model is loaded only once for this test.
from app.models.loader import load_model


# Import the request schema.
# This represents the data that would normally come
# from an API client.
from app.schemas.request import GenerationRequest


# Import the service layer that connects the request
# to the model inference function.
from app.services.generation_service import GenerationService


def main():

    # ---------------------------------------------------------
    # STEP 1: Load the model
    # ---------------------------------------------------------
    #
    # In the production application, this will happen when
    # the application starts rather than for every request.
    #
    print("Loading model...")

    tokenizer, model = load_model()

    print("Model loaded.")


    # ---------------------------------------------------------
    # STEP 2: Create the generation service
    # ---------------------------------------------------------
    #
    # The service receives the already-loaded model and tokenizer.
    #
    # The service will reuse them for generation requests.
    #
    service = GenerationService(
        tokenizer=tokenizer,
        model=model,
    )


    # ---------------------------------------------------------
    # STEP 3: Create a test request
    # ---------------------------------------------------------
    #
    # This mimics the request that an API client will eventually
    # send to our FastAPI endpoint.
    #
    request = GenerationRequest(
        instruction="Solve the following reasoning problem.",
        input="""
Quinton is looking to add 4 fruit trees to his backyard.  He wants to plant 2 apple trees that will be 10 feet wide each and need 12 feet between them.  The peach trees will be closer to the house and will grow to be 12 feet wide and will need 15 feet between each tree.  All told, how much space will these trees take up in his yard?
""",
        max_new_tokens=256,
    )


    # ---------------------------------------------------------
    # STEP 4: Send the request to the service
    # ---------------------------------------------------------
    #
    # The service will:
    #
    # GenerationRequest
    #       ↓
    # GenerationService
    #       ↓
    # generate_response()
    #       ↓
    # Llama model
    #
    result = service.generate(request)


    # ---------------------------------------------------------
    # STEP 5: Display the model response
    # ---------------------------------------------------------
    #
    # result is a GenerationResponse object.
    # Its generated text is stored in result.response.
    #
    print("\nResponse:")
    print(result.response)


# This ensures main() runs only when this file
# is executed directly.
if __name__ == "__main__":
    main()