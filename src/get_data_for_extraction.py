import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input dataset (JSON format).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output JSON file.",
    )
    return parser.parse_args()


system_prompt = """You are a highly skilled assistant specializing in extracting relevant information from provided documents. Your task is to identify and extract sentences from the documents as much as possible that are most directly useful for answering the given question. Rank the sentences in order of relevance, with the most relevant sentence listed first. Preface each sentence with its sequence number as follows:
Sentence 1: 
......
Sentence n:
"""


def main():
    args = parse_args()
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f]

    final_data = []

    for item in data:
        docs = "\n".join(
            [f"document {j}:{doc}" for j, doc in enumerate(item["documents"], start=1)]
        )
        sentences = "\n".join(
            [
                f"sentence {j}:{sentence}"
                for j, sentence in enumerate(item["sentences"], start=1)
            ]
        )
        final_data.append(
            {
                "instruction": system_prompt,
                "input": f"Question:{item['question']}\nDocuments:{docs}",
                "output": sentences,
                "system": "",
                "history": [],
            }
        )

    print(len(final_data))
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=4)


if __name__ == "__main__":
    main()
