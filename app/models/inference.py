import torch


def generate_response(
    instruction: str,
    user_input: str,
    tokenizer,
    model,
    max_new_tokens: int = 512,
):
    """
    Generate a response from the fine-tuned Llama model.

    The prompt format follows the chat format used during fine-tuning.
    """

    user_content = f"{instruction}\n\n{user_input}".strip()

    messages = [
        {
            "role": "user",
            "content": user_content,
        }
    ]

    # Apply the same chat template used by the Llama tokenizer.
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    # With device_map="auto", use the device of the model's
    # input embedding layer rather than model.device.
    input_device = model.get_input_embeddings().weight.device

    inputs = {
        key: value.to(input_device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Remove the original prompt tokens.
    input_length = inputs["input_ids"].shape[1]

    generated_tokens = outputs[0][input_length:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response.strip()