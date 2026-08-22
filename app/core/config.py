import os


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

TORCH_DTYPE = os.getenv(
    "TORCH_DTYPE",
    "float16",
)

DEVICE_MAP = os.getenv(
    "DEVICE_MAP",
    "auto",
)