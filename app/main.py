from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.models.loader import load_model
from app.services.generation_service import GenerationService

import logging
import time
from app.core.logging import setup_logging

# Configure application logging.
setup_logging()

#  logger for this module.
logger = logging.getLogger(__name__)


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
    logger.info("Loading model...")

    tokenizer, model = load_model()

    logger.info("Model loaded successfully.")

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

    logger.info("Generation service initialized.")

    # Application is now ready to receive requests.
    yield

    # ---------------------------------------------------------
    # APPLICATION SHUTDOWN
    # ---------------------------------------------------------

    logger.info("Shutting down application...")

    tokenizer = None
    model = None
    generation_service = None

    logger.info("Application shutdown complete.")


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


@app.get("/ready")
def readiness_check():
    """
    Check whether the application is ready to serve
    model-generation requests.
    """

    generation_service = getattr(
        app.state,
        "generation_service",
        None,
    )

    if (
        generation_service is None
        or model is None
        or tokenizer is None
    ):
        return {
            "status": "not_ready",
            "model_loaded": False,
        }

    return {
        "status": "ready",
        "model_loaded": True,
    }

  
@app.middleware("http")
async def log_request_latency(request, call_next):
    """
    Measure the total time taken to process an HTTP request.

    This includes:
    - FastAPI request handling
    - validation
    - service execution
    - model inference
    - response creation
    """

    start_time = time.perf_counter()

    response = await call_next(request)

    latency = time.perf_counter() - start_time

    logger.info(
        "Request %s %s completed with status=%s in %.3f seconds.",
        request.method,
        request.url.path,
        response.status_code,
        latency,
    )

    return response