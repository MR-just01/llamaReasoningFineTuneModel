import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER = "MR023/Llama3.2-Reasoning"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER,
    )

    model.eval()

    return tokenizer, model