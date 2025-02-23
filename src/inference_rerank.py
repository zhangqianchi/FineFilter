import argparse
import os
import torch
from sentence_transformers import SentenceTransformer, util
import json
from tqdm import tqdm


def rank_documents(query, documents, model, top_k=None, save_results=False):
    query_embedding = model.encode(query, convert_to_tensor=True)
    document_embeddings = model.encode(documents, convert_to_tensor=True)
    cosine_similarities = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
    sorted_indices = cosine_similarities.argsort(descending=True)

    if top_k:
        sorted_indices = sorted_indices[:top_k]

    sorted_results = [
        {"document": documents[idx], "score": float(cosine_similarities[idx])}
        for idx in sorted_indices
    ]

    if save_results:
        with open("sorted_inference_results.json", "w") as f:
            json.dump(sorted_results, f, indent=4)
        print("\nResults saved to 'sorted_inference_results.json'")

    return sorted_results


def clean_and_split_sentences(input_string):
    sentences = input_string.split("\n")
    cleaned_sentences = [
        sentence.split(": ", 1)[1] if ": " in sentence else sentence
        for sentence in sentences
    ]
    return cleaned_sentences


def load_qa_documents(file_path):
    question, answer, documents, summary = [], [], [], []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                data = json.loads(line)
                question.append(data["question"])
                answer.append(data["answer"])
                documents.append(data["documents"])
                summary.append(clean_and_split_sentences(data["summary"]))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")

    return question, answer, documents, summary


def main():
    parser = argparse.ArgumentParser(
        description="Document ranking and summarization with bi-encoder."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pre-trained SentenceTransformer model",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input Q&A documents (JSON format)",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save the ranked output"
    )
    parser.add_argument(
        "--gpu_device",
        type=str,
        default="0",
        help="Specify the GPU device to use (default: 0)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Specify the number of top-ranked documents to return",
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Whether to save the results to a file",
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(args.model_path, device=device)
    print(f"Model loaded successfully on {device}")

    question, answer, documents, summary = load_qa_documents(args.input_path)

    for i in tqdm(range(len(question))):
        new_summary = rank_documents(
            query=question[i],
            documents=summary[i],
            model=model,
            top_k=args.top_k,
            save_results=args.save_results,
        )
        if summary[i] != new_summary:
            print("yes")
        summary[i] = new_summary

        with open(args.output_path, "a+", encoding="utf-8") as f:
            res = {
                "question": question[i],
                "answer": answer[i],
                "documents": documents[i],
                "summary": new_summary,
            }
            json.dump(res, f)
            f.write("\n")


if __name__ == "__main__":
    main()
