from app.models.loader import load_model
from app.models.inference import generate_response


def main():
    print("Loading model...")

    tokenizer, model = load_model()

    print("Model loaded.")

    prompt = "What is 2 + 2?"

    response = generate_response(
        prompt=prompt,
        tokenizer=tokenizer,
        model=model,
    )

    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()