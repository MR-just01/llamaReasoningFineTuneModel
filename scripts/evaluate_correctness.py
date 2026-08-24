import os
import re
import math
import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

# Results produced by Phase 7B.
INPUT_PATH = (
    "results/model_evaluation_results.csv"
)

# Final correctness evaluation file.
OUTPUT_PATH = (
    "results/model_correctness_results.csv"
)


# =========================================================
# LOAD RESULTS
# =========================================================

print("=" * 70)
print("MODEL ANSWER CORRECTNESS EVALUATION")
print("=" * 70)

df = pd.read_csv(INPUT_PATH)

print(
    f"Rows loaded: {len(df)}"
)


# =========================================================
# ANSWER EXTRACTION
# =========================================================

def extract_final_answer(response):
    """
    Extract the model's final answer.

    Preferred format:

        Final Answer:
        42

    If Final Answer is missing, fall back to:

        Answer:
        42
    """

    if pd.isna(response):
        return ""

    response = str(response).strip()

    # -----------------------------------------------------
    # Try "Final Answer:"
    # -----------------------------------------------------

    match = re.search(
        r"Final\s*Answer\s*:\s*(.+)",
        response,
        flags=re.IGNORECASE,
    )

    if match:
        answer = match.group(1).strip()

        # Only keep the first line.
        answer = answer.splitlines()[0].strip()

        return answer


    # -----------------------------------------------------
    # Fallback: "Answer:"
    # -----------------------------------------------------

    match = re.search(
        r"\bAnswer\s*:\s*(.+)",
        response,
        flags=re.IGNORECASE,
    )

    if match:
        answer = match.group(1).strip()

        answer = answer.splitlines()[0].strip()

        return answer


    # -----------------------------------------------------
    # No answer marker found.
    # -----------------------------------------------------

    return ""


# =========================================================
# NORMALIZE ANSWERS
# =========================================================

def normalize_answer(answer):
    """
    Normalize answers before comparison.

    Examples:

        520,000  -> 520000
        520000   -> 520000

        YES      -> yes
        Yes      -> yes
    """

    if answer is None:
        return ""

    answer = str(answer).strip().lower()

    # Remove surrounding whitespace.
    answer = answer.strip()

    # Remove commas from numbers.
    answer = answer.replace(",", "")

    # Remove trailing punctuation.
    answer = answer.rstrip(".;:")

    # Collapse multiple spaces.
    answer = re.sub(
        r"\s+",
        " ",
        answer,
    )

    return answer


# =========================================================
# NUMERIC COMPARISON
# =========================================================

def try_parse_number(value):
    """
    Try to convert a simple numeric answer into a float.

    Handles examples such as:

        42
        42.0
        -5
        520000
        12 min
        0.45%
    """

    if not value:
        return None

    value = normalize_answer(value)

    # -----------------------------------------------------
    # Extract first numeric value.
    # -----------------------------------------------------

    match = re.search(
        r"-?\d+(?:\.\d+)?",
        value,
    )

    if not match:
        return None

    try:

        return float(
            match.group(0)
        )

    except ValueError:

        return None


# =========================================================
# ANSWER COMPARISON
# =========================================================

def answers_match(
    expected,
    predicted,
):
    """
    Compare expected and predicted answers.

    Strategy:

    1. Exact normalized comparison.
    2. Numeric comparison.
    3. Otherwise mark as incorrect.
    """

    expected_norm = normalize_answer(
        expected
    )

    predicted_norm = normalize_answer(
        predicted
    )

    # -----------------------------------------------------
    # Empty prediction = incorrect.
    # -----------------------------------------------------

    if not predicted_norm:
        return False


    # -----------------------------------------------------
    # Exact normalized match.
    # -----------------------------------------------------

    if expected_norm == predicted_norm:
        return True


    # -----------------------------------------------------
    # Numeric comparison.
    # -----------------------------------------------------

    expected_number = try_parse_number(
        expected_norm
    )

    predicted_number = try_parse_number(
        predicted_norm
    )

    if (
        expected_number is not None
        and predicted_number is not None
    ):

        return math.isclose(
            expected_number,
            predicted_number,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )


    # -----------------------------------------------------
    # No match.
    # -----------------------------------------------------

    return False


# =========================================================
# EVALUATE EACH RESPONSE
# =========================================================

print()
print("Evaluating model responses...")

results = []


for _, row in df.iterrows():

    model_response = row[
        "model_response"
    ]

    expected_answer = row[
        "expected_answer"
    ]


    # -----------------------------------------------------
    # Extract answer from model response.
    # -----------------------------------------------------

    predicted_answer = (
        extract_final_answer(
            model_response
        )
    )


    # -----------------------------------------------------
    # Compare against audited ground truth.
    # -----------------------------------------------------

    answer_correct = answers_match(
        expected_answer,
        predicted_answer,
    )


    # -----------------------------------------------------
    # Check whether the model hit the generation limit.
    # -----------------------------------------------------

    generation_truncated = bool(
        row.get(
            "generation_truncated",
            False,
        )
    )


    # -----------------------------------------------------
    # Preserve all existing information.
    # -----------------------------------------------------

    result = row.to_dict()


    result.update(
        {
            "predicted_answer": (
                predicted_answer
            ),

            "normalized_expected_answer": (
                normalize_answer(
                    expected_answer
                )
            ),

            "normalized_predicted_answer": (
                normalize_answer(
                    predicted_answer
                )
            ),

            "answer_correct": (
                answer_correct
            ),

            "generation_truncated": (
                generation_truncated
            ),
        }
    )


    results.append(result)


# =========================================================
# CREATE RESULTS DATAFRAME
# =========================================================

results_df = pd.DataFrame(
    results
)


# =========================================================
# SAVE RESULTS
# =========================================================

os.makedirs(
    "results",
    exist_ok=True,
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


# =========================================================
# CALCULATE ACCURACY
# =========================================================

total_rows = len(
    results_df
)

correct_count = int(
    results_df[
        "answer_correct"
    ].sum()
)

incorrect_count = (
    total_rows
    - correct_count
)

accuracy = (
    correct_count
    / total_rows
) * 100


# =========================================================
# TRUNCATION STATISTICS
# =========================================================

truncated_count = int(
    results_df[
        "generation_truncated"
    ].sum()
)

truncation_rate = (
    truncated_count
    / total_rows
) * 100


# =========================================================
# API SUCCESS STATISTICS
# =========================================================

successful_count = int(
    results_df[
        "successful"
    ].sum()
)

failed_count = (
    total_rows
    - successful_count
)

success_rate = (
    successful_count
    / total_rows
) * 100


# =========================================================
# PERFORMANCE METRICS
# =========================================================

successful_rows = results_df[
    results_df["successful"] == True
]


if len(successful_rows) > 0:

    mean_latency = (
        successful_rows[
            "latency_ms"
        ].mean()
    )

    p50_latency = (
        successful_rows[
            "latency_ms"
        ].quantile(0.50)
    )

    p95_latency = (
        successful_rows[
            "latency_ms"
        ].quantile(0.95)
    )

    mean_output_tokens = (
        successful_rows[
            "output_tokens"
        ].mean()
    )

    mean_tokens_per_second = (
        successful_rows[
            "tokens_per_second"
        ].mean()
    )

else:

    mean_latency = 0
    p50_latency = 0
    p95_latency = 0
    mean_output_tokens = 0
    mean_tokens_per_second = 0


# =========================================================
# PRINT REPORT
# =========================================================

print()
print("=" * 70)
print("MODEL CORRECTNESS EVALUATION RESULTS")
print("=" * 70)

print(
    f"Total evaluated:       {total_rows}"
)

print(
    f"Correct answers:       {correct_count}"
)

print(
    f"Incorrect answers:     {incorrect_count}"
)

print(
    f"Answer accuracy:       {accuracy:.2f}%"
)

print()

print(
    f"API successful:        {successful_count}"
)

print(
    f"API failed:            {failed_count}"
)

print(
    f"API success rate:      {success_rate:.2f}%"
)

print()

print(
    f"Truncated responses:   {truncated_count}"
)

print(
    f"Truncation rate:       {truncation_rate:.2f}%"
)

print()

print(
    f"Mean latency:          "
    f"{mean_latency:.2f} ms"
)

print(
    f"P50 latency:           "
    f"{p50_latency:.2f} ms"
)

print(
    f"P95 latency:           "
    f"{p95_latency:.2f} ms"
)

print(
    f"Mean output tokens:    "
    f"{mean_output_tokens:.2f}"
)

print(
    f"Mean tokens/sec:       "
    f"{mean_tokens_per_second:.2f}"
)

print()

print(
    f"Results saved to: "
    f"{OUTPUT_PATH}"
)

print("=" * 70)