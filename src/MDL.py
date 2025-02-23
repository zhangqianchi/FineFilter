import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json


def compute_average_mdl_score(model, query, sentence, tokenizer):
    input_text = f"{query}\n{sentence}"
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    sentence_start_idx = inputs.input_ids.size(1) - len(
        tokenizer(sentence)["input_ids"]
    )
    logits = logits[:, sentence_start_idx - 1 : -1, :]
    labels = inputs.input_ids[:, sentence_start_idx:].contiguous()

    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

    num_tokens = labels.size(1)
    average_mdl = total_loss.item() / num_tokens

    return average_mdl


def rerank_with_average_mdl(model, query, sentences, tokenizer):
    scores = [
        (sentence, compute_average_mdl_score(model, query, sentence, tokenizer))
        for sentence in sentences
    ]
    sorted_sentences = sorted(scores, key=lambda x: x[1])
    return sorted_sentences


def clean_and_split_sentences(input_string):
    sentences = input_string.split("\n")
    cleaned_sentences = [
        sentence.split(":", 1)[1] if ":" in sentence else sentence
        for sentence in sentences
    ]
    return cleaned_sentences


def main():
    parser = argparse.ArgumentParser(
        description="Re-rank sentences using LLaMA-3 with MDL scoring."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pre-trained LLaMA model",
    )
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the input JSON file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the reranked results",
    )
    parser.add_argument(
        "--gpu_device",
        type=str,
        default="1",
        help="Specify the GPU device to use (default: 1)",
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to("cuda").eval()

    with open(args.input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f]

    for item in data:
        sentences = clean_and_split_sentences(item["summary"])
        sorted_sentences = rerank_with_average_mdl(
            model, item["question"], sentences, tokenizer
        )

        with open(args.output_path, "a+", encoding="utf-8") as f:
            res = {
                "question": item["question"],
                "answer": item["answer"],
                "documents": item["documents"],
                "summary": item["summary"],
                "rerank": sorted_sentences,
            }
            json.dump(res, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    main()
