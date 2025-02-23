import os
import re
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sentence_transformers import SentenceTransformer


# Argument parsing function
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the pre-trained model."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the dataset (JSON format).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output results.",
    )
    parser.add_argument(
        "--similarity_threshold",
        default=0.25,
        type=float,
        help="Cosine similarity threshold for filtering.",
    )
    parser.add_argument(
        "--batch_size", default=32, type=int, help="Batch size for processing data."
    )
    return parser.parse_args()


# Load the pre-trained model and tokenizer
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    return tokenizer, model, device


# Encode the texts using the model and tokenizer
def encode_texts(texts, tokenizer, model, device):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embeddings


# Filter sentences based on cosine similarity
def filter_sentences_by_similarity(a, b, threshold, tokenizer, model, device):
    a_embeddings = encode_texts(a, tokenizer, model, device)
    b_embeddings = encode_texts(b, tokenizer, model, device)

    distances = cosine_distances(a_embeddings, b_embeddings)
    neighbors = set()

    for i, a_sentence in enumerate(a):
        for j, b_sentence in enumerate(b):
            if distances[i, j] <= threshold:
                neighbors.add(b_sentence)

    return list(neighbors)


# Split paragraphs into sentences
def split_into_sentences(paragraph):
    return re.split(r"(?<=[.!?]) +", paragraph)


# Find relevant sentences based on the answer
def finding_sentence(answer, sentences):
    result = set()
    for ans in answer:
        for sentence in sentences:
            if ans in sentence:
                result.add(sentence)
    return list(result)


# Sort sentences based on similarity to the query
def sorted_sentences(query, sentences, model1):
    e_q = model1.encode([query])
    embeddings = model1.encode(sentences)
    similarities = cosine_similarity(e_q, embeddings)[0]
    sorted_sentences = sorted(
        zip(sentences, similarities), key=lambda x: x[1], reverse=True
    )
    return [sentence for sentence, _ in sorted_sentences]


# Main function to process the data and save results
def main():
    # Parse arguments
    args = parse_args()

    # Load the model and tokenizer
    tokenizer, model, device = load_model(args.model_path)

    # Initialize SentenceTransformer for sorting
    model1 = SentenceTransformer(args.model_path)

    # Open the input data file
    with open(args.data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f]

    all_number = 0
    number = 0

    for item in data:
        answer = item["answer"]
        documents = item["documents"]
        sentences = []

        # Split documents into sentences
        for doc in documents:
            sentences.extend(split_into_sentences(doc))

        result = finding_sentence(answer, sentences)

        if len(result) == 0:
            continue
        else:
            all_number += 1
            filtered_results = filter_sentences_by_similarity(
                result,
                sentences,
                threshold=args.similarity_threshold,
                tokenizer=tokenizer,
                model=model,
                device=device,
            )

            if len(result) != len(filtered_results):
                number += 1
            # Sort the filtered sentences based on similarity to the query
            final_filtered_result = sorted_sentences(
                query=item["question"], sentences=filtered_results, model1=model1
            )
            # Save the results to the output file
            with open(args.output_path, "a+", encoding="utf-8") as f:
                res = {
                    "question": item["question"],
                    "answer": item["answer"],
                    "documents": item["documents"],
                    "sentences": final_filtered_result,
                }
                json.dump(res, f)
                f.write("\n")


if __name__ == "__main__":
    main()
