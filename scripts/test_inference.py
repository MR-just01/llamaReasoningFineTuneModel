from app.models.loader import load_model
from app.models.inference import generate_response


def main():

    print("Loading model...")

    tokenizer, model = load_model()

    print("Model loaded.")

    instruction = "Solve the following reasoning problem."

    user_input = """
If a train travels 60 kilometers in 1 hour,
how far will it travel in 3 hours at the same speed?
"""

    print("\nGenerating response...")

    response = generate_response(
        instruction=instruction,
        user_input=user_input,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=256,
    )

    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()