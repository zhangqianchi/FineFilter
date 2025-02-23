import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
import transformers
from tqdm import tqdm
import random
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with fine-tuned Llama model."
    )
    parser.add_argument("--cuda_device", type=str, default="1", help="CUDA device id.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the fine-tuned model."
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the input data file."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output results.",
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Number of top tokens to sample."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.75,
        help="Cumulative probability for token sampling.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.01, help="Sampling temperature."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=9000,
        help="Maximum length of generated sequences.",
    )

    return parser.parse_args()


def load_llama(model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = base_model.eval()
    return model, tokenizer


def load_qa_documents(data_path):
    question = []
    answer = []
    documents = []
    with open(data_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                data = json.loads(line)
                question.append(data["question"])
                answer.append(data["answer"])
                documents.append(data["summary"])
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")

    return question, answer, documents


def clean_and_split_sentences(input_string):
    sentences = input_string.split("\n")
    cleaned_sentences = [
        sentence.split(": ", 1)[1] if ": " in sentence else sentence
        for sentence in sentences
    ]
    return cleaned_sentences


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    question, answer, documents = load_qa_documents(args.data_path)
    model, tokenizer = load_llama(args.model_path)

    number = 0

    for i in tqdm(range(len(question))):
        cleaned_sentences = clean_and_split_sentences(documents[i])
        final_docs = "\n".join(
            [
                f"sentence {j}:{item}"
                for j, item in enumerate(cleaned_sentences, start=1)
            ]
        )

        system_prompt = f"You are a helpful, respectful and honest assistant, and please use documents provided to answer the query. Documents:\n{final_docs}"
        input_text = (
            f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{question[i]} [/INST]"
        )

        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        sequences = pipeline(
            input_text,
            do_sample=True,
            top_k=args.top_k,
            top_p=args.top_p,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=args.max_length,
            return_full_text=False,
            temperature=args.temperature,
        )

        ans = sequences[0]["generated_text"]
        with open(args.output_path, "a+", encoding="utf-8") as f:
            res = {"question": question[i], "answer": ans}
            json.dump(res, f)
            f.write("\n")
