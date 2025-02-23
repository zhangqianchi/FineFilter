import sys
import json
import logging
import os
import argparse
import pandas as pd
import random
from datetime import datetime
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,
    LoggingHandler,
    models,
    evaluation,
    losses,
    InputExample,
)
from sklearn.model_selection import train_test_split

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Configure logging
def configure_logging():
    file_handler = logging.FileHandler("./rerank.log", mode="a")
    file_handler.setLevel(logging.INFO)
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler(), file_handler],
    )


# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the input data (JSONL format).",
    )
    parser.add_argument(
        "--train_batch_size", default=64, type=int, help="Batch size for training."
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="Maximum sequence length for input.",
    )
    parser.add_argument(
        "--model_name",
        default="your_model_path",
        type=str,
        help="Pre-trained model name or path.",
    )
    parser.add_argument(
        "--epochs", default=64, type=int, help="Number of epochs to train."
    )
    parser.add_argument(
        "--pooling",
        default="mean",
        type=str,
        choices=["mean", "max", "cls"],
        help="Pooling strategy.",
    )
    parser.add_argument(
        "--warmup_steps", default=1000, type=int, help="Number of warmup steps."
    )
    parser.add_argument("--lr", default=2e-5, type=float, help="Learning rate.")
    parser.add_argument(
        "--checkpoint_save_total_limit",
        default=3,
        type=int,
        help="Total number of checkpoints to keep.",
    )
    parser.add_argument(
        "--eval_steps",
        default=100,
        type=int,
        help="Number of steps between evaluations.",
    )
    return parser.parse_args()


# Model initialization function
def initialize_model(args):
    logging.info("Initializing SentenceTransformer model...")
    word_embedding_model = models.Transformer(
        args.model_name, max_seq_length=args.max_seq_length
    )
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), pooling_mode=args.pooling
    )
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model


# Custom dataset class
class CustomDataset:
    def __init__(self, json_file_path):
        self.data_df = []
        with open(json_file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                question = data["question"]
                positive_texts = data.get("positive", [])
                negative_texts = []

                if data["negative_id"]:
                    for neg_idx in data["negative_id"]:
                        if neg_idx < len(data["documents"]):
                            negative_texts.append(data["documents"][neg_idx])

                if positive_texts and negative_texts:
                    for pos_text in positive_texts:
                        for neg_text in negative_texts:
                            self.data_df.append((question, pos_text, neg_text))

        logging.info(f"Loaded {len(self.data_df)} total triplets from {json_file_path}")

    def split_data(self, train_size=0.8):
        train_data, dev_data = train_test_split(
            self.data_df, train_size=train_size, random_state=42
        )
        return train_data, dev_data


# Main logic
def main():
    # Configure logging
    configure_logging()

    # Parse command-line arguments
    args = parse_args()
    logging.info(f"Arguments: {args}")

    # Initialize the model
    model = initialize_model(args)

    # Create model save path
    model_save_path = f'output/train_bi-encoder-mnrl-{args.model_name.replace("/", "-")}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    # Load and split dataset
    logging.info("Loading and splitting dataset...")
    dataset = CustomDataset(args.data_path)
    train_data, dev_data = dataset.split_data()

    # Create DataLoader for training and validation
    train_dataset = [
        InputExample(texts=[query, pos, neg]) for query, pos, neg in train_data
    ]
    dev_dataset = [
        InputExample(texts=[query, pos, neg]) for query, pos, neg in dev_data
    ]
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.train_batch_size
    )
    dev_dataloader = DataLoader(
        dev_dataset, shuffle=False, batch_size=args.train_batch_size
    )

    # Define training loss function
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Define evaluation evaluator
    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(dev_dataset)

    # Start training
    logging.info("Starting training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        use_amp=True,
        checkpoint_path=model_save_path,
        evaluation_steps=args.eval_steps,
        checkpoint_save_total_limit=args.checkpoint_save_total_limit,
        optimizer_params={"lr": args.lr},
        save_best_model=True,
    )

    # Save the final model
    model.save(model_save_path)
    logging.info(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main()
