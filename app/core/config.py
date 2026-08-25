import os

from dotenv import load_dotenv


# Load variables from .env when running locally.
# In production, environment variables can be
# supplied directly by the deployment environment.
load_dotenv()


# =========================================================
# MODEL CONFIGURATION
# =========================================================

MODEL_ID = os.getenv(
    "MODEL_ID",
    "meta-llama/Llama-3.2-3B-Instruct",
)

ADAPTER_ID = os.getenv(
    "ADAPTER_ID",
    "MR023/Llama3.2-Reasoning",
)

MAX_NEW_TOKENS = int(
    os.getenv(
        "MAX_NEW_TOKENS",
        "512",
    )
)

HF_TOKEN = os.getenv("HF_TOKEN")

TORCH_DTYPE = os.getenv(
    "TORCH_DTYPE",
    "float16",
)

DEVICE_MAP = os.getenv(
    "DEVICE_MAP",
    "auto",
)


# =========================================================
# API CONFIGURATION
# =========================================================

HOST = os.getenv(
    "HOST",
    "0.0.0.0",
)

PORT = int(
    os.getenv(
        "PORT",
        "8000",
    )
)