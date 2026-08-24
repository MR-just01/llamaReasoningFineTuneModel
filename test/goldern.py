import pandas as pd

df = pd.read_csv(
    "data/answer_correct_166_audit_progress (2).csv"
)

golden_df = df[
    (df["question_valid"] == True) &
    (df["reference_answer_correct"] == True)
]

print("Total audited rows:", len(df))
print("Golden evaluation rows:", len(golden_df))
print()
print(golden_df[
    [
        "index",
        "input",
        "expected_answer",
        "manual_answer_correct",
        "manual_reasoning_correct",
        "audit_status",
    ]
].head())