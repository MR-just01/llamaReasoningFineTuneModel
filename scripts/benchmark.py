import time
import statistics
import torch

from app.models.loader import load_model
from app.models.inference import generate_response


def synchronize_gpu():
    """Wait for GPU operations to finish before measuring."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def percentile(values, p):
    """Calculate percentile using linear interpolation."""
    values = sorted(values)

    if not values:
        return 0.0

    if len(values) == 1:
        return values[0]

    k = (len(values) - 1) * (p / 100)
    lower = int(k)
    upper = min(lower + 1, len(values) - 1)

    weight = k - lower

    return values[lower] + weight * (values[upper] - values[lower])


def main():

    print("Loading model...")

    tokenizer, model = load_model()

    print("Model loaded.")

    instruction = (
        "Solve the following multiple-choice reasoning problem."
    )

    prompts = [
        """
A farmer has 17 sheep. All but 9 die. How many sheep are left?

A) 8
B) 9
C) 17
D) 26
""",
        """
If all roses are flowers and some flowers fade quickly,
can we conclude that all roses fade quickly?

A) Yes
B) No
C) Only red roses
D) Cannot determine
""",
        """
A train travels 60 km in 1 hour. How far will it travel
in 3 hours at the same speed?

A) 120 km
B) 180 km
C) 200 km
D) 240 km
""",
        """
There are 5 boxes. Each box contains 4 balls.
How many balls are there in total?

A) 9
B) 15
C) 20
D) 25
""",
        """
John is taller than Mike. Mike is taller than David.
Who is the shortest?

A) John
B) Mike
C) David
D) Cannot determine
""",
    ]

    # --------------------------------------------------
    # Warm-up
    # --------------------------------------------------

    print("\nRunning warm-up...")

    for _ in range(2):
        generate_response(
            instruction=instruction,
            user_input=prompts[0],
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=256,
        )

    synchronize_gpu()

    print("Warm-up complete.")

    # --------------------------------------------------
    # Benchmark
    # --------------------------------------------------

    print("\nStarting benchmark...")

    latencies = []

    for i, prompt in enumerate(prompts, start=1):

        print(f"Running request {i}/{len(prompts)}...")

        synchronize_gpu()

        start_time = time.perf_counter()

        response = generate_response(
            instruction=instruction,
            user_input=prompt,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=256,
        )

        synchronize_gpu()

        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000

        latencies.append(latency_ms)

        print(f"Latency: {latency_ms:.2f} ms")

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    mean_latency = statistics.mean(latencies)
    median_latency = statistics.median(latencies)

    p95_latency = percentile(latencies, 95)
    p99_latency = percentile(latencies, 99)

    min_latency = min(latencies)
    max_latency = max(latencies)

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)

    print(f"Requests:        {len(latencies)}")
    print(f"Mean latency:    {mean_latency:.2f} ms")
    print(f"P50 latency:     {median_latency:.2f} ms")
    print(f"P95 latency:     {p95_latency:.2f} ms")
    print(f"P99 latency:     {p99_latency:.2f} ms")
    print(f"Min latency:     {min_latency:.2f} ms")
    print(f"Max latency:     {max_latency:.2f} ms")

    print("=" * 50)


if __name__ == "__main__":
    main()