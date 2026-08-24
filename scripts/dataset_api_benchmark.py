import time
import pandas as pd
import requests


# =========================================================
# CONFIGURATION
# =========================================================

# Path to the dataset containing the reasoning questions.
DATASET_PATH = "data/reasoning_questions.csv"

# FastAPI endpoint that serves our fine-tuned model.
API_URL = "http://127.0.0.1:8000/generate"

# ---------------------------------------------------------
# IMPORTANT:
# Start with 5 rows while testing the benchmark.
# Once everything works, change this to 50.
# ---------------------------------------------------------
NUM_ROWS = 5

# Maximum time allowed for one API request.
REQUEST_TIMEOUT = 120

# Maximum number of tokens the model can generate.
MAX_NEW_TOKENS = 256

# File where individual benchmark results will be saved.
OUTPUT_PATH = "data/dataset_api_benchmark_results.csv"


# =========================================================
# LOAD DATASET
# =========================================================

print("=" * 60)
print("DATASET-BASED API LATENCY BENCHMARK")
print("=" * 60)

print(f"Dataset: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)

print(f"Total rows available: {len(df)}")


# Make sure the dataset contains the required column.
if "question" not in df.columns:
    raise ValueError(
        "Dataset must contain a 'question' column."
    )


# ---------------------------------------------------------
# Select rows for the benchmark.
# ---------------------------------------------------------

benchmark_df = df.head(NUM_ROWS).copy()

print(
    f"Rows selected for benchmark: "
    f"{len(benchmark_df)}"
)

print()


# =========================================================
# RUN BENCHMARK
# =========================================================

results = []


for position, (_, row) in enumerate(
    benchmark_df.iterrows(),
    start=1,
):

    question_id = row["id"]
    question = row["question"]

    print("-" * 60)
    print(
        f"Request {position}/{len(benchmark_df)}"
    )
    print(f"Question ID: {question_id}")

    # -----------------------------------------------------
    # Build request payload.
    # -----------------------------------------------------
    #
    # The question comes directly from the dataset.
    #
    payload = {
        "instruction": (
            "Solve the following reasoning problem."
        ),
        "input": question,
        "max_new_tokens": MAX_NEW_TOKENS,
    }

    # Default values in case the request fails.
    status_code = None
    model_response = ""
    error_message = ""
    successful = False

    # -----------------------------------------------------
    # START API LATENCY TIMER
    # -----------------------------------------------------
    #
    # This measures the complete HTTP round trip:
    #
    # Python client
    #      ↓
    # HTTP request
    #      ↓
    # FastAPI
    #      ↓
    # model inference
    #      ↓
    # FastAPI response
    #      ↓
    # Python client
    #
    start_time = time.perf_counter()

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

        # Stop the timer immediately after receiving
        # the HTTP response.
        end_time = time.perf_counter()

        latency_ms = (
            end_time - start_time
        ) * 1000

        status_code = response.status_code

        # -------------------------------------------------
        # SUCCESSFUL REQUEST
        # -------------------------------------------------

        if response.status_code == 200:

            successful = True

            response_json = response.json()

            model_response = response_json.get(
                "response",
                "",
            )

            print(
                f"Latency: {latency_ms:.2f} ms"
            )

            print("Status: 200 OK")

        # -------------------------------------------------
        # FAILED HTTP REQUEST
        # -------------------------------------------------

        else:

            error_message = response.text

            print(
                f"Request failed: "
                f"HTTP {response.status_code}"
            )

    # -----------------------------------------------------
    # REQUEST / CONNECTION / TIMEOUT ERROR
    # -----------------------------------------------------

    except requests.RequestException as exc:

        end_time = time.perf_counter()

        latency_ms = (
            end_time - start_time
        ) * 1000

        error_message = str(exc)

        print(
            f"Request failed: {error_message}"
        )

    # =====================================================
    # SAVE RESULT FOR THIS DATASET ROW
    # =====================================================

    results.append(
        {
            # Dataset information
            "id": question_id,
            "category": row.get(
                "category",
                "",
            ),
            "difficulty": row.get(
                "difficulty",
                "",
            ),
            "reasoning_type": row.get(
                "reasoning_type",
                "",
            ),
            "question": question,
            "expected_answer": row.get(
                "expected_answer",
                "",
            ),

            # Model/API output
            "model_response": model_response,

            # Performance information
            "latency_ms": latency_ms,
            "status_code": status_code,
            "successful": successful,

            # Error information
            "error": error_message,
        }
    )


# =========================================================
# CREATE RESULTS DATAFRAME
# =========================================================

results_df = pd.DataFrame(results)


# =========================================================
# SAVE INDIVIDUAL RESULTS
# =========================================================

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
)

print()
print(
    f"Individual results saved to: "
    f"{OUTPUT_PATH}"
)


# =========================================================
# CALCULATE LATENCY METRICS
# =========================================================

successful_latencies = results_df.loc[
    results_df["successful"],
    "latency_ms",
]


if len(successful_latencies) > 0:

    # -----------------------------------------------------
    # Calculate overall API latency statistics.
    # -----------------------------------------------------

    mean_latency = (
        successful_latencies.mean()
    )

    p50_latency = (
        successful_latencies.quantile(0.50)
    )

    p95_latency = (
        successful_latencies.quantile(0.95)
    )

    p99_latency = (
        successful_latencies.quantile(0.99)
    )

    min_latency = (
        successful_latencies.min()
    )

    max_latency = (
        successful_latencies.max()
    )

    successful_count = (
        results_df["successful"].sum()
    )

    failed_count = (
        len(results_df) - successful_count
    )

    success_rate = (
        successful_count
        / len(results_df)
    ) * 100

    error_rate = (
        failed_count
        / len(results_df)
    ) * 100

    # -----------------------------------------------------
    # Print final benchmark report.
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("DATASET API BENCHMARK RESULTS")
    print("=" * 60)

    print(
        f"Total requests:     {len(results_df)}"
    )

    print(
        f"Successful:         {successful_count}"
    )

    print(
        f"Failed:             {failed_count}"
    )

    print(
        f"Success rate:       {success_rate:.2f}%"
    )

    print(
        f"Error rate:         {error_rate:.2f}%"
    )

    print()

    print(
        f"Mean latency:       "
        f"{mean_latency:.2f} ms"
    )

    print(
        f"P50 latency:        "
        f"{p50_latency:.2f} ms"
    )

    print(
        f"P95 latency:        "
        f"{p95_latency:.2f} ms"
    )

    print(
        f"P99 latency:        "
        f"{p99_latency:.2f} ms"
    )

    print(
        f"Min latency:        "
        f"{min_latency:.2f} ms"
    )

    print(
        f"Max latency:        "
        f"{max_latency:.2f} ms"
    )

    print("=" * 60)


else:

    print()
    print(
        "No successful requests were recorded."
    )