import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from app.core.config import (
    MODEL_ID,
    ADAPTER_ID,
    TORCH_DTYPE,
    DEVICE_MAP,
)


def get_torch_dtype(dtype_name: str):
    """Convert a configuration string into a PyTorch dtype."""

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }

    if dtype_name not in dtype_map:
        raise ValueError(
            f"Unsupported TORCH_DTYPE: {dtype_name}. "
            f"Supported values: {list(dtype_map.keys())}"
        )

    return dtype_map[dtype_name]


def load_model():
    dtype = get_torch_dtype(TORCH_DTYPE)

    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_ID
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        device_map=DEVICE_MAP,
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_ID,
    )

    model.eval()

    return tokenizer, model