"""CCS Inference — score individual questions or batches."""

import os
import sys
import torch
import numpy as np
from dataclasses import dataclass
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ccs_model.config import (
    MAX_SEQ_LEN, GPU_DEVICE, OUTPUT_DIR, BLOOM_LEVELS
)
from ccs_model.model import CCSModel
from ccs_model.data_loader import get_tokenizer, format_input_text


@dataclass
class CCSScore:
    level: int          # 1-6
    name: str           # "Remember", "Analyze", etc.
    confidence: float   # how close to integer (regression) or class probability (classification)
    raw_score: float    # raw model output


class CCSPredictor:
    """Load a trained CCS model and score questions."""

    def __init__(self, checkpoint_path=None, mode="regression", device=None):
        if device is None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_DEVICE)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.mode = mode
        self.tokenizer = get_tokenizer()

        # Load full model (encoder + head)
        self.model = CCSModel(mode=mode)

        if checkpoint_path is None:
            checkpoint_path = os.path.join(OUTPUT_DIR, f"fold_0/best_model_{mode}.pt")

        if os.path.exists(checkpoint_path):
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"WARNING: No checkpoint at {checkpoint_path}. Using untrained model.")

        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, question_text: str, subject: str = "general") -> CCSScore:
        """Score a single question."""
        question = {"question_text": question_text, "subject": subject}
        text = format_input_text(question)

        encoding = self.tokenizer(
            text, max_length=MAX_SEQ_LEN, padding="max_length",
            truncation=True, return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        token_type_ids = encoding.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        output = self.model(input_ids, attention_mask, token_type_ids)

        if self.mode == "regression":
            raw_score = output.item()
            level = int(np.clip(round(raw_score), 1, 6))
            confidence = 1.0 - abs(raw_score - level)
        else:
            probs = torch.softmax(output, dim=1)[0]
            level = probs.argmax().item() + 1
            confidence = probs[level - 1].item()
            raw_score = (probs.cpu().numpy() * np.arange(1, 7)).sum()

        return CCSScore(level=level, name=BLOOM_LEVELS[level],
                        confidence=float(confidence), raw_score=float(raw_score))

    @torch.no_grad()
    def predict_batch(self, questions: List[dict], batch_size: int = 32) -> List[CCSScore]:
        """Score a batch of questions. Each dict needs 'question_text' and optionally 'subject'."""
        results = []

        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            texts = [format_input_text(q) for q in batch]

            encoding = self.tokenizer(
                texts, max_length=MAX_SEQ_LEN, padding="max_length",
                truncation=True, return_tensors="pt"
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            token_type_ids = encoding.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            output = self.model(input_ids, attention_mask, token_type_ids)

            if self.mode == "regression":
                scores = output.cpu().numpy()
                for s in scores:
                    level = int(np.clip(round(s), 1, 6))
                    confidence = 1.0 - abs(s - level)
                    results.append(CCSScore(level, BLOOM_LEVELS[level], float(confidence), float(s)))
            else:
                probs = torch.softmax(output, dim=1).cpu().numpy()
                for p in probs:
                    level = p.argmax() + 1
                    confidence = p[level - 1]
                    raw_score = (p * np.arange(1, 7)).sum()
                    results.append(CCSScore(level, BLOOM_LEVELS[level], float(confidence), float(raw_score)))

        return results


def predict(question_text: str, subject: str = "general", mode: str = "regression") -> CCSScore:
    """Quick-predict a single question (loads model on first call)."""
    if not hasattr(predict, "_predictor") or predict._predictor.mode != mode:
        predict._predictor = CCSPredictor(mode=mode)
    return predict._predictor.predict(question_text, subject)


if __name__ == "__main__":
    predictor = CCSPredictor(mode="regression")

    test_questions = [
        ("What is the chemical formula for water?", "chemistry"),
        ("Explain how osmosis works in your own words.", "biology"),
        ("Calculate the pH of a 0.1M HCl solution.", "chemistry"),
        ("Compare and contrast mitosis and meiosis.", "biology"),
        ("Evaluate whether the author's argument is supported by evidence.", "general"),
        ("Design an experiment to test whether fertilizer affects plant growth.", "biology"),
    ]

    print("CCS Predictions:")
    print("-" * 70)
    for text, subject in test_questions:
        score = predictor.predict(text, subject)
        print(f"  L{score.level} ({score.name:10s}) | conf={score.confidence:.3f} | "
              f"raw={score.raw_score:.2f} | {text[:60]}")
