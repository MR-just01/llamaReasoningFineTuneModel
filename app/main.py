from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.models.loader import load_model
from app.services.generation_service import GenerationService


# These variables will hold the model components
# after the application starts.
tokenizer = None
model = None
generation_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the lifecycle of the application.

    The model is loaded once when the application starts
    and remains in memory while the application is running.

    This is important because loading Llama + LoRA is expensive.
    We do NOT want to load the model for every API request.
    """

    global tokenizer, model, generation_service

    # ---------------------------------------------------------
    # APPLICATION STARTUP
    # ---------------------------------------------------------

    # Load the tokenizer and fine-tuned model once.
    print("Loading model...")

    tokenizer, model = load_model()

    print("Model loaded successfully.")

    # Create the service once.
    #
    # The service will reuse the same model for every request.
    generation_service = GenerationService(
        tokenizer=tokenizer,
        model=model,
    )

    # Store the service inside FastAPI's application state.
    # API routes can access this same service for every request.
    app.state.generation_service = generation_service

    print("Generation service initialized.")

    # Application is now ready to receive requests.
    yield

    # ---------------------------------------------------------
    # APPLICATION SHUTDOWN
    # ---------------------------------------------------------

    print("Shutting down application...")

    tokenizer = None
    model = None
    generation_service = None

    print("Application shutdown complete.")


# ---------------------------------------------------------
# CREATE FASTAPI APPLICATION
# ---------------------------------------------------------

app = FastAPI(
    title="Llama 3.2 Reasoning API",
    description="API for the fine-tuned Llama 3.2 reasoning model.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------
# REGISTER API ROUTES
# ---------------------------------------------------------
#
# This connects the router from routes.py to our FastAPI app.
#
# Without this, /generate will return:
#
#     404 Not Found
#
app.include_router(router)


# ---------------------------------------------------------
# HEALTH CHECK ENDPOINT
# ---------------------------------------------------------
#
# This endpoint allows us to check whether the API is running.
#
# It does NOT run the model.
#
@app.get("/health")
def health_check():
    """
    Check whether the application is running.
    """

    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }