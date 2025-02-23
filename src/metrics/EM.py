import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dev_file",
        type=str,
        required=True,
        help="Path to the dev file (JSON format).",
    )
    parser.add_argument(
        "--baseline_file",
        type=str,
        required=True,
        help="Path to the baseline file (JSON format).",
    )
    return parser.parse_args()


def compute_exact_match(predictions, references):
    exact_matches = []
    for pred, refs in zip(predictions, references):
        if pred in refs:
            exact_matches.append(1)
        else:
            exact_matches.append(0)
    return sum(exact_matches) / len(exact_matches) * 100


def load_answers(file_path):
    df = pd.read_json(file_path, lines=True)
    return df["answer"].tolist()


def main():
    args = parse_args()

    given_answer = load_answers(args.dev_file)
    generated_answer = load_answers(args.baseline_file)

    exact_match_score = compute_exact_match(
        predictions=generated_answer, references=given_answer
    )

    print(f"Exact Match Score: {exact_match_score:.4f}%")


if __name__ == "__main__":
    main()
