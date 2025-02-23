import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
from tqdm import tqdm


def load_llama3(model_id):
    """Load the LLaMA model and corresponding tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
    )
    # Ensure pad_token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def get_summary(query, docs, tokenizer, model):
    """Generate a summary based on the question and documents"""
    system_prompt = """You are a highly skilled assistant specializing in extracting relevant information from provided documents. Your task is to identify and extract sentences from the documents as much as possible that are most directly useful for answering the given question. Rank the sentences in order of relevance, with the most relevant sentence listed first. Preface each sentence with its sequence number as follows:
    Sentence 1: 
    ......
    Sentence n:
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {query}\nDocuments: {docs}"},
    ]

    # Build the input
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    # Check the input structure
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    # Text generation
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.01,
        top_p=0.6,
    )

    # Decode the generated response
    response = outputs[0][input_ids.shape[-1] :]
    print(tokenizer.decode(response, skip_special_tokens=True))
    return tokenizer.decode(response, skip_special_tokens=True)


def load_qa_documents(file_path):
    """Load Q&A documents"""
    question, answer, documents = [], [], []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                data = json.loads(line)
                question.append(data["question"])
                answer.append(data["answer"])
                documents.append(data["documents"])
            except json.JSONDecodeError as e:
                print(f"JSON parse error: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")
    return question, answer, documents


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Summarization with LLaMA model.")
    parser.add_argument(
        "--model_id", type=str, required=True, help="Path to the LLaMA model"
    )
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the input Q&A documents"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the summarized output",
    )
    parser.add_argument(
        "--gpu_device",
        type=str,
        default="0",
        help="Specify the GPU device to use (default: 0)",
    )

    args = parser.parse_args()

    # Set the available GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    tokenizer, model = load_llama3(args.model_id)
    question, answer, documents = load_qa_documents(args.input_path)

    for i in tqdm(range(len(question))):
        try:
            # Concatenate documents
            docs = "\n".join(
                [
                    f"document {j}: {item}"
                    for j, item in enumerate(documents[i], start=1)
                ]
            )
            summary = get_summary(question[i], docs, tokenizer, model)

            # Write to file
            with open(args.output_path, "a+", encoding="utf-8") as f:
                res = {
                    "question": question[i],
                    "answer": answer[i],
                    "documents": documents[i],
                    "summary": summary,
                }
                json.dump(res, f, ensure_ascii=False)
                f.write("\n")

        except Exception as e:
            print(f"An error occurred while processing question {i}: {e}")


if __name__ == "__main__":
    main()
