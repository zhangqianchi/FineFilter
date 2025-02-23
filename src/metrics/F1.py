import re
from collections import Counter
import pandas as pd


def normalize_text(text: str) -> str:
    """Normalize text with lowercasing, removing articles, and punctuation."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(lower(text)))


def calc_unigram_f1(text: str, answers: list[str], field: str = "f1") -> float:
    """Calculate unigram f1 score between the text and reference answers."""
    norm_pred = normalize_text(text)
    norm_answers = [normalize_text(ans) for ans in answers]
    common_tokens = [
        Counter(norm_pred) & Counter(norm_ans) for norm_ans in norm_answers
    ]
    num_same = [sum(common.values()) for common in common_tokens]

    score_list = []
    for i, num in enumerate(num_same):
        if num == 0:
            score_list.append(0.0)
        else:
            p = 1.0 * num / len(norm_pred)
            r = 1.0 * num / len(norm_answers[i])
            f1 = 2 * p * r / (p + r)
            if field == "precision":
                score_list.append(p)
            elif field == "recall":
                score_list.append(r)
            elif field == "f1":
                score_list.append(f1)
            else:
                raise ValueError(f"Unknown field: {field}")
    return max(score_list)


def cal_F1(generated_answer, given_answer):
    scores = []
    for text, answers in zip(generated_answer, given_answer):
        scores.append(calc_unigram_f1(text, answers))
    print(sum(scores) / len(scores))
    return sum(scores) / len(scores)
