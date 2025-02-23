import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import transformers
from tqdm import tqdm


def load_llama(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = base_model.eval()
    return model, tokenizer


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
    parser = argparse.ArgumentParser(description="Run QA inference using Llama model.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned Llama model",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input Q&A documents (JSON format)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the inference results",
    )
    parser.add_argument(
        "--gpu_device",
        type=str,
        default="0",
        help="Specify the GPU device to use (default: 0)",
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    model, tokenizer = load_llama(args.model_path)
    question, answer, documents, summary = load_qa_documents(args.input_path)

    for i in tqdm(range(len(question))):
        positive, negative, positive_id, negative_id = [], [], [], []
        for sentence in summary[i]:
            final_docs = f"document 1: {sentence}"
            system_prompt = f"You are a helpful, respectful and honest assistant, and please use documents provided to answer the query. Documents:\n{final_docs}"
            input = (
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
                input,
                do_sample=True,
                top_k=5,
                top_p=0.75,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=9000,
                return_full_text=False,
                temperature=0.01,
            )
            ans = sequences[0]["generated_text"]
            if ans.strip() in answer[i]:
                positive.append(sentence)
                positive_id.append(summary[i].index(sentence))
            else:
                negative.append(sentence)
                negative_id.append(summary[i].index(sentence))

        with open(args.output_path, "a+", encoding="utf-8") as f:
            res = {
                "question": question[i],
                "answer": answer[i],
                "documents": documents[i],
                "summary": summary[i],
                "positive": positive,
                "positive_id": positive_id,
                "negative": negative,
                "negative_id": negative_id,
            }
            json.dump(res, f)
            f.write("\n")


if __name__ == "__main__":
    main()
