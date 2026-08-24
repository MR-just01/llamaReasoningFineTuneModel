import os
import time

import pandas as pd
import requests


# =========================================================
# CONFIGURATION
# =========================================================

# ---------------------------------------------------------
# Golden evaluation dataset
# ---------------------------------------------------------
#
# This is the human-audited dataset.
# The expected_answer is used as ground truth AFTER
# the model generates its response.
#
GOLDEN_DATASET_PATH = (
    "data/answer_correct_166_audit_progress (2).csv"
)


# ---------------------------------------------------------
# FastAPI generation endpoint
# ---------------------------------------------------------

API_URL = (
    "http://127.0.0.1:8000/generate"
)


# ---------------------------------------------------------
# Number of rows to evaluate
# ---------------------------------------------------------
#
# Start with 5 rows.
#
# After verifying everything works:
#
# NUM_ROWS = 166
#
NUM_ROWS = 5


# ---------------------------------------------------------
# API configuration
# ---------------------------------------------------------

MAX_NEW_TOKENS = 256

REQUEST_TIMEOUT = 120


# ---------------------------------------------------------
# Output file
# ---------------------------------------------------------

OUTPUT_PATH = ("results/model_evaluation_results.csv")


# =========================================================
# LOAD GOLDEN DATASET
# =========================================================

print("=" * 70)
print("MODEL QUALITY EVALUATION")
print("=" * 70)

print(
    f"Dataset: {GOLDEN_DATASET_PATH}")
df = pd.read_csv(
    GOLDEN_DATASET_PATH)
print( f"Total audited rows: {len(df)}")


# =========================================================
# VERIFY GOLDEN DATASET
# =========================================================
#
# We only want rows that were confirmed by the human
# audit as valid and having a correct reference answer.
#

golden_df = df[
    (df["question_valid"] == True)
    &
    (df["reference_answer_correct"] == True)
].copy()


print(
    f"Golden evaluation rows available: "
    f"{len(golden_df)}")


# ---------------------------------------------------------
# Verify that the requested number of rows is available.
# ---------------------------------------------------------

if NUM_ROWS > len(golden_df):

    raise ValueError(
        f"Requested {NUM_ROWS} rows, "
        f"but only {len(golden_df)} "
        f"golden rows are available.")

# ---------------------------------------------------------
# Select rows for this evaluation run.
# ---------------------------------------------------------

evaluation_df = (golden_df.head(NUM_ROWS).copy())


print(
    f"Rows selected for evaluation: "
    f"{len(evaluation_df)}")

print()
# =========================================================
# CREATE OUTPUT DIRECTORY
# =========================================================

os.makedirs(
    "results",
    exist_ok=True,
)


# =========================================================
# RUN MODEL EVALUATION
# =========================================================

results = []


for position, (_, row) in enumerate(
    evaluation_df.iterrows(),
    start=1,
):

    print("-" * 70)

    print(
        f"Evaluation {position}/"
        f"{len(evaluation_df)}"
    )

    print(
        f"Question ID: "
        f"{row.get('id', row.get('index', ''))}"
    )


    # =====================================================
    # READ DATASET INFORMATION
    # =====================================================

    question = row["input"]

    expected_answer = row[
        "expected_answer"
    ]


    # =====================================================
    # BUILD API REQUEST
    # =====================================================
    #
    # IMPORTANT:
    #
    # expected_answer is NOT sent to the model.
    #
    # The model receives only the question.
    #
    # expected_answer is retained locally as the
    # ground-truth reference for later evaluation.
    #

    payload = {

        "instruction": (
            "Solve the following reasoning problem."
        ),

        "input": question,

        "max_new_tokens": MAX_NEW_TOKENS,
    }


    # =====================================================
    # DEFAULT VALUES
    # =====================================================

    model_response = ""

    status_code = None

    input_tokens = 0

    output_tokens = 0

    total_tokens = 0

    tokens_per_second = 0.0

    latency_ms = 0.0

    successful = False

    error_message = ""


    # =====================================================
    # START LATENCY TIMER
    # =====================================================

    start_time = time.perf_counter()


    try:

        # -------------------------------------------------
        # Send request to FastAPI.
        # -------------------------------------------------

        response = requests.post(
            API_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )


        # -------------------------------------------------
        # Stop timer immediately after receiving response.
        # -------------------------------------------------

        end_time = time.perf_counter()


        latency_ms = (
            end_time - start_time
        ) * 1000


        status_code = response.status_code


        # =================================================
        # SUCCESSFUL RESPONSE
        # =================================================

        if response.status_code == 200:

            successful = True


            response_json = (
                response.json()
            )


            # -------------------------------------------------
            # Generated model response
            # -------------------------------------------------

            model_response = (
                response_json.get(
                    "response",
                    "",
                )
            )


            # -------------------------------------------------
            # Token metrics returned by the API
            # -------------------------------------------------

            input_tokens = (
                response_json.get(
                    "input_tokens",
                    0,
                )
            )


            output_tokens = (
                response_json.get(
                    "output_tokens",
                    0,
                )
            )


            total_tokens = (
                response_json.get(
                    "total_tokens",
                    0,
                )
            )


            # -------------------------------------------------
            # Calculate output throughput.
            # -------------------------------------------------

            if (
                output_tokens > 0
                and latency_ms > 0
            ):

                tokens_per_second = (
                    output_tokens
                    / (latency_ms / 1000)
                )


            print(
                f"Status: 200 OK"
            )

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


        # =================================================
        # FAILED HTTP RESPONSE
        # =================================================

        else:

            error_message = (
                response.text
            )


            print(
                f"Request failed: "
                f"HTTP {response.status_code}"
            )


    # =====================================================
    # CONNECTION / TIMEOUT ERROR
    # =====================================================

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
    # STORE RESULT
    # =====================================================

    results.append(
        {

            # -------------------------------------------------
            # Dataset information
            # -------------------------------------------------

            "id": row.get(
                "id",
                row.get("index", ""),
            ),

            "question": question,

            "expected_answer": expected_answer,


            # -------------------------------------------------
            # New model response
            # -------------------------------------------------

            "model_response": model_response,


            # -------------------------------------------------
            # Token metrics
            # -------------------------------------------------

            "input_tokens": input_tokens,

            "output_tokens": output_tokens,

            "total_tokens": total_tokens,


            # -------------------------------------------------
            # Performance metrics
            # -------------------------------------------------

            "latency_ms": latency_ms,

            "tokens_per_second": (
                tokens_per_second
            ),


            # -------------------------------------------------
            # Request status
            # -------------------------------------------------

            "status_code": status_code,

            "successful": successful,

            "error": error_message,
        }
    )


# =========================================================
# CREATE RESULTS DATAFRAME
# =========================================================

results_df = pd.DataFrame(
    results
)


# =========================================================
# SAVE RESULTS
# =========================================================

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


print()

print(
    f"Evaluation results saved to: "
    f"{OUTPUT_PATH}"
)


# =========================================================
# EVALUATION SUMMARY
# =========================================================

successful_results = (
    results_df[
        results_df["successful"]
    ]
)


successful_count = len(
    successful_results
)


failed_count = (
    len(results_df)
    - successful_count
)


print()
print("=" * 70)
print("EVALUATION RUN SUMMARY")
print("=" * 70)


print(
    f"Total requests:     "
    f"{len(results_df)}"
)


print(
    f"Successful:         "
    f"{successful_count}"
)


print(
    f"Failed:             "
    f"{failed_count}"
)


if len(results_df) > 0:

    success_rate = (
        successful_count
        / len(results_df)
    ) * 100


    print(
        f"Success rate:       "
        f"{success_rate:.2f}%"
    )


# ---------------------------------------------------------
# Performance summary
# ---------------------------------------------------------

if successful_count > 0:

    mean_latency = (
        successful_results[
            "latency_ms"
        ].mean()
    )


    p50_latency = (
        successful_results[
            "latency_ms"
        ].quantile(0.50)
    )


    p95_latency = (
        successful_results[
            "latency_ms"
        ].quantile(0.95)
    )


    mean_output_tokens = (
        successful_results[
            "output_tokens"
        ].mean()
    )


    mean_tokens_per_second = (
        successful_results[
            "tokens_per_second"
        ].mean()
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
        f"Mean output tokens: "
        f"{mean_output_tokens:.2f}"
    )


    print(
        f"Mean tokens/sec:    "
        f"{mean_tokens_per_second:.2f}"
    )


print("=" * 70)