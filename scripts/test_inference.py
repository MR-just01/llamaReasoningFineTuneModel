import time
import torch

from app.models.loader import load_model
from app.models.inference import generate_response


def synchronize_gpu():
    """Wait for GPU operations to finish before measuring time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():

    print("Loading model...")

    tokenizer, model = load_model()

    print("Model loaded.")

    instruction = "Solve the following multiple-choice reasoning problem."

    instruction = "Solve the following multiple-choice reasoning problem."

    user_input = """
A farmer has 17 sheep. All but 9 die. How many sheep are left?

A) 8
B) 9
C) 17
D) 26
"""

    print("\nWarming up the model...")

    # Warm-up run.
    # We don't include this in the latency measurement because
    # the first generation can include one-time GPU initialization.
    generate_response(
        instruction=instruction,
        user_input=user_input,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=256,
    )

    synchronize_gpu()

    print("Warm-up complete.")

    print("\nGenerating response...")

    # Start latency measurement
    synchronize_gpu()
    start_time = time.perf_counter()

    response = generate_response(
        instruction=instruction,
        user_input=user_input,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=256,
    )

    synchronize_gpu()
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000

    print("\nResponse:")
    print(response)

    print("\n--- Performance ---")
    print(f"Inference latency: {latency_ms:.2f} ms")
    print(f"Inference latency: {latency_ms / 1000:.2f} seconds")


if __name__ == "__main__":
    main()