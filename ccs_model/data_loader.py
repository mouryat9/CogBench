"""Data loading and preprocessing for CCS model training."""

import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold
import numpy as np

from ccs_model.config import (
    DATA_FILE, MODEL_NAME, MAX_SEQ_LEN, N_FOLDS, SEED, NUM_LEVELS
)


def load_labeled_questions(data_file=None):
    """Load questions that have bloom_level labels (739 from Yahya + Mohammed)."""
    data_file = data_file or DATA_FILE
    with open(data_file, encoding="utf-8") as f:
        all_questions = json.load(f)

    labeled = [q for q in all_questions if q.get("bloom_level") is not None]
    print(f"Loaded {len(labeled)} labeled questions from {len(all_questions)} total")

    # Print distribution
    from collections import Counter
    dist = Counter(q["bloom_level"] for q in labeled)
    for level in sorted(dist):
        print(f"  L{level}: {dist[level]}")

    return labeled


def format_input_text(question):
    """Format question text with subject prefix for DeBERTa input."""
    subject = question.get("subject", "general")
    text = question["question_text"]
    return f"[{subject}] {text}"


class BloomDataset(Dataset):
    """PyTorch dataset for Bloom's Taxonomy classification."""

    def __init__(self, questions, tokenizer, max_len=MAX_SEQ_LEN, mode="regression"):
        self.questions = questions
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode

        self.texts = [format_input_text(q) for q in questions]
        self.labels = [q["bloom_level"] for q in questions]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        label = self.labels[idx]
        if self.mode == "regression":
            item["label"] = torch.tensor(float(label), dtype=torch.float32)
        else:
            item["label"] = torch.tensor(label - 1, dtype=torch.long)  # 0-indexed for CE loss

        return item


def get_kfold_splits(questions, n_folds=N_FOLDS, seed=SEED):
    """Generate stratified K-fold splits preserving Bloom's level distribution."""
    labels = [q["bloom_level"] for q in questions]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    splits = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(questions, labels)):
        train_qs = [questions[i] for i in train_idx]
        val_qs = [questions[i] for i in val_idx]
        splits.append({
            "fold": fold_idx,
            "train": train_qs,
            "val": val_qs,
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
        })

    return splits


def get_tokenizer():
    """Load DeBERTa tokenizer."""
    return AutoTokenizer.from_pretrained(MODEL_NAME)


if __name__ == "__main__":
    # Quick test
    questions = load_labeled_questions()
    tokenizer = get_tokenizer()
    splits = get_kfold_splits(questions)

    for split in splits:
        train_ds = BloomDataset(split["train"], tokenizer)
        val_ds = BloomDataset(split["val"], tokenizer)
        print(f"Fold {split['fold']}: train={len(train_ds)}, val={len(val_ds)}")

    # Test one item
    item = train_ds[0]
    print(f"\nSample item keys: {list(item.keys())}")
    print(f"input_ids shape: {item['input_ids'].shape}")
    print(f"label: {item['label']}")
