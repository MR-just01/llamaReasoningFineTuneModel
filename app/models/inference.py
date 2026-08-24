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

    The prompt format follows the chat format used during
    fine-tuning.

    Returns:
        A dictionary containing:
        - generated response
        - input token count
        - output token count
        - total token count
    """

    # ---------------------------------------------------------
    # BUILD USER PROMPT
    # ---------------------------------------------------------

    user_content = f"{instruction}\n\n{user_input}".strip()

    messages = [
        {
            "role": "user",
            "content": user_content,
        }
    ]

    # ---------------------------------------------------------
    # TOKENIZE INPUT
    # ---------------------------------------------------------
    #
    # Apply the same chat template used during fine-tuning.
    #
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    # ---------------------------------------------------------
    # COUNT INPUT TOKENS
    # ---------------------------------------------------------
    #
    # input_ids contains the complete prompt after applying
    # the chat template.
    #
    input_tokens = inputs["input_ids"].shape[1]

    # ---------------------------------------------------------
    # MOVE INPUT TO MODEL DEVICE
    # ---------------------------------------------------------
    #
    # With device_map="auto", use the device of the model's
    # input embedding layer rather than model.device.
    #
    input_device = model.get_input_embeddings().weight.device

    inputs = {
        key: value.to(input_device)
        for key, value in inputs.items()
    }

    # ---------------------------------------------------------
    # GENERATE RESPONSE
    # ---------------------------------------------------------

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # ---------------------------------------------------------
    # COUNT GENERATED TOKENS
    # ---------------------------------------------------------
    #
    # outputs contains:
    #
    # [original input tokens] + [new generated tokens]
    #
    # Therefore, subtract the input length from the total
    # sequence length to get the number of newly generated
    # tokens.
    #
    total_tokens = outputs[0].shape[0]

    output_tokens = total_tokens - input_tokens

    # ---------------------------------------------------------
    # EXTRACT ONLY GENERATED TOKENS
    # ---------------------------------------------------------
    #
    # Remove the original prompt tokens before decoding.
    #
    generated_tokens = outputs[0][input_tokens:]

    # ---------------------------------------------------------
    # DECODE GENERATED TOKENS
    # ---------------------------------------------------------

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # ---------------------------------------------------------
    # RETURN GENERATION + TOKEN METRICS
    # ---------------------------------------------------------

    return {
        "response": response.strip(),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }