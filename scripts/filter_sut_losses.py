"""Filter test-zd.csv to rows where sut_win == 0.0, extracting the first
question and first baseline_output into a simplified CSV."""

import json
from pathlib import Path

import pandas as pd

INPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "test-zd.csv"


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    # Parse first question from inputs JSON
    df["question"] = df["inputs"].apply(
        lambda x: json.loads(x).get("question", [""])[0]
    )

    # Parse first baseline_output from reference_outputs JSON
    df["answer"] = df["reference_outputs"].apply(
        lambda x: json.loads(x).get("baseline_output", [""])[0]
    )

    # Replace newlines so each row stays on one line
    df["question"] = df["question"].str.replace("\n", " ", regex=False)
    df["answer"] = df["answer"].str.replace("\n", " ", regex=False)

    # Filter rows where sut_win is 0.0 and add sequential id
    losses = df[df["sut_win"] == 0.0][["question", "answer"]].reset_index(drop=True)
    losses.insert(0, "id", losses.index + 1)

    output_path = INPUT_PATH.with_name("test-zd-sut-losses.csv")
    losses.to_csv(output_path, index=False)
    print(f"Saved {len(losses)} rows to {output_path}")


if __name__ == "__main__":
    main()
