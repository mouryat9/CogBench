"""CCS Model — BERT-base-uncased with full end-to-end fine-tuning.

BERT-base is numerically stable for fine-tuning (unlike DeBERTa-v3 which
has NaN issues in its disentangled attention during backprop). We use
differential learning rates: lower LR for the pretrained encoder, higher
LR for the classifier head.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

from ccs_model.config import (
    MODEL_NAME, HIDDEN_DIM, DROPOUT, MIN_LEVEL, MAX_LEVEL, NUM_LEVELS
)


class CCSModel(nn.Module):
    """BERT encoder + classifier head for Bloom's level prediction.

    Supports two modes:
    - regression: sigmoid output scaled to [1.0, 6.0]
    - classification: 6-class logits
    """

    def __init__(self, model_name=MODEL_NAME, hidden_dim=HIDDEN_DIM,
                 dropout=DROPOUT, mode="regression"):
        super().__init__()
        self.mode = mode
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size  # 768 for bert-base

        output_dim = 1 if mode == "regression" else NUM_LEVELS
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        logits = self.head(cls_output)

        if self.mode == "regression":
            logits = logits.squeeze(-1)
            return torch.sigmoid(logits) * (MAX_LEVEL - MIN_LEVEL) + MIN_LEVEL
        return logits

    def get_optimizer_groups(self, encoder_lr, head_lr, weight_decay=0.01):
        """Return parameter groups with differential learning rates."""
        no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}

        encoder_decay = []
        encoder_no_decay = []
        head_decay = []
        head_no_decay = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("encoder."):
                if any(nd in name for nd in no_decay):
                    encoder_no_decay.append(param)
                else:
                    encoder_decay.append(param)
            else:
                if any(nd in name for nd in no_decay):
                    head_no_decay.append(param)
                else:
                    head_decay.append(param)

        return [
            {"params": encoder_decay, "lr": encoder_lr, "weight_decay": weight_decay},
            {"params": encoder_no_decay, "lr": encoder_lr, "weight_decay": 0.0},
            {"params": head_decay, "lr": head_lr, "weight_decay": weight_decay},
            {"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0},
        ]


def count_parameters(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    print("Testing CCSModel (regression)...")
    model = CCSModel(mode="regression")
    t, a = count_parameters(model)
    print(f"Params: {t:,} trainable / {a:,} total")

    dummy_ids = torch.randint(0, 1000, (2, 128)).cuda()
    dummy_mask = torch.ones(2, 128, dtype=torch.long).cuda()
    model = model.cuda()
    scores = model(dummy_ids, dummy_mask)
    print(f"Regression scores: {scores.tolist()}")

    print("\nTesting CCSModel (classification)...")
    model_cls = CCSModel(mode="classification").cuda()
    logits = model_cls(dummy_ids, dummy_mask)
    preds = logits.argmax(dim=1) + 1
    print(f"Predicted levels: {preds.tolist()}")

    print("\nTesting optimizer groups...")
    groups = model.get_optimizer_groups(encoder_lr=2e-5, head_lr=1e-4)
    for i, g in enumerate(groups):
        n_params = sum(p.numel() for p in g["params"])
        print(f"  Group {i}: {n_params:,} params, lr={g['lr']}, wd={g['weight_decay']}")
