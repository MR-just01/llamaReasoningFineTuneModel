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

# Start with 5 rows while testing.
# Once everything works, change this to 50.
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


# Select the rows for the benchmark.
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
    # BUILD API REQUEST
    # -----------------------------------------------------

    payload = {
        "instruction": (
            "Solve the following reasoning problem."
        ),
        "input": question,
        "max_new_tokens": MAX_NEW_TOKENS,
    }

    # -----------------------------------------------------
    # DEFAULT VALUES
    # -----------------------------------------------------
    #
    # These values are used if the request fails.
    #

    status_code = None
    model_response = ""

    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    tokens_per_second = 0.0

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

        # Stop timer immediately after receiving
        # the HTTP response.
        end_time = time.perf_counter()

        latency_ms = (
            end_time - start_time
        ) * 1000

        status_code = response.status_code

        # SUCCESSFUL REQUEST
        if response.status_code == 200:
            successful = True
            response_json = response.json()
            # Generated model response.
            model_response = response_json.get("response","",)
            # Token usage returned by the API.
            input_tokens = response_json.get("input_tokens",0, )
            output_tokens = response_json.get(  "output_tokens", 0, )
            total_tokens = response_json.get( "total_tokens",0,  )

            # -------------------------------------------------
            # CALCULATE OUTPUT TOKENS / SECOND
            # -------------------------------------------------
            #
            # This measures how many generated output tokens
            # were produced per second from the API client's
            # perspective.
            #

            if output_tokens > 0 and latency_ms > 0:

                tokens_per_second = (
                    output_tokens
                    / (latency_ms / 1000)
                )

            else:

                tokens_per_second = 0.0

            print(
                f"Latency: "
                f"{latency_ms:.2f} ms"
            )

            print(
                f"Input tokens: "
                f"{input_tokens}"
            )

            print(
                f"Output tokens: "
                f"{output_tokens}"
            )

            print(
                f"Total tokens: "
                f"{total_tokens}"
            )

            print(
                f"Tokens/sec: "
                f"{tokens_per_second:.2f}"
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
    # CONNECTION / TIMEOUT ERROR
    # -----------------------------------------------------

    except requests.RequestException as exc:

        end_time = time.perf_counter()

        latency_ms = (
            end_time - start_time
        ) * 1000

        error_message = str(exc)

        print(
            f"Request failed: "
            f"{error_message}"
        )

    # =====================================================
    # SAVE RESULT FOR THIS DATASET ROW
    # =====================================================

    results.append(
        {
            # -------------------------------------------------
            # DATASET INFORMATION
            # -------------------------------------------------

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

            # -------------------------------------------------
            # MODEL OUTPUT
            # -------------------------------------------------

            "model_response": model_response,

            # -------------------------------------------------
            # TOKEN METRICS
            # -------------------------------------------------

            "input_tokens": input_tokens,

            "output_tokens": output_tokens,

            "total_tokens": total_tokens,

            # -------------------------------------------------
            # PERFORMANCE METRICS
            # -------------------------------------------------

            "latency_ms": latency_ms,

            "tokens_per_second": tokens_per_second,

            # -------------------------------------------------
            # REQUEST STATUS
            # -------------------------------------------------

            "status_code": status_code,

            "successful": successful,

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

successful_results = results_df[
    results_df["successful"]
]


if len(successful_results) > 0:

    # -----------------------------------------------------
    # LATENCY SERIES
    # -----------------------------------------------------

    successful_latencies = (
        successful_results["latency_ms"]
    )

    # -----------------------------------------------------
    # TOKEN/S SECOND SERIES
    # -----------------------------------------------------

    successful_throughput = (
        successful_results[
            "tokens_per_second"
        ]
    )

    # -----------------------------------------------------
    # LATENCY STATISTICS
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

    # -----------------------------------------------------
    # TOKEN STATISTICS
    # -----------------------------------------------------

    mean_input_tokens = (
        successful_results[
            "input_tokens"
        ].mean()
    )

    mean_output_tokens = (
        successful_results[
            "output_tokens"
        ].mean()
    )

    mean_total_tokens = (
        successful_results[
            "total_tokens"
        ].mean()
    )

    mean_tokens_per_second = (
        successful_throughput.mean()
    )

    # -----------------------------------------------------
    # REQUEST STATISTICS
    # -----------------------------------------------------

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
    # PRINT FINAL BENCHMARK REPORT
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("DATASET API BENCHMARK RESULTS")
    print("=" * 60)

    print(
        f"Total requests:       "
        f"{len(results_df)}"
    )

    print(
        f"Successful:           "
        f"{successful_count}"
    )

    print(
        f"Failed:               "
        f"{failed_count}"
    )

    print(
        f"Success rate:         "
        f"{success_rate:.2f}%"
    )

    print(
        f"Error rate:           "
        f"{error_rate:.2f}%"
    )

    print()

    print(
        f"Mean latency:         "
        f"{mean_latency:.2f} ms"
    )

    print(
        f"P50 latency:          "
        f"{p50_latency:.2f} ms"
    )

    print(
        f"P95 latency:          "
        f"{p95_latency:.2f} ms"
    )

    print(
        f"P99 latency:          "
        f"{p99_latency:.2f} ms"
    )

    print(
        f"Min latency:          "
        f"{min_latency:.2f} ms"
    )

    print(
        f"Max latency:          "
        f"{max_latency:.2f} ms"
    )

    print()

    print(
        f"Mean input tokens:    "
        f"{mean_input_tokens:.2f}"
    )

    print(
        f"Mean output tokens:   "
        f"{mean_output_tokens:.2f}"
    )

    print(
        f"Mean total tokens:    "
        f"{mean_total_tokens:.2f}"
    )

    print(
        f"Mean tokens/sec:      "
        f"{mean_tokens_per_second:.2f}"
    )

    print("=" * 60)


else:

    print()
    print(
        "No successful API requests were recorded."
    )