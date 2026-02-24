"""CCS Model configuration — all hyperparameters in one place."""

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "all_collected_questions.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ccs_model", "checkpoints", "ccs_deberta_v1")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

# Model — BERT-base-uncased (110M params, 768 hidden, 12 layers)
# Numerically stable for full fine-tuning (unlike DeBERTa-v3-large which has NaN issues)
MODEL_NAME = "bert-base-uncased"
MAX_SEQ_LEN = 128       # Questions are short (~20-30 tokens avg)
HIDDEN_DIM = 256        # Classifier head hidden dimension
DROPOUT = 0.1

# Training
BATCH_SIZE = 16
LEARNING_RATE = 2e-5    # Encoder LR
HEAD_LR_MULTIPLIER = 5  # Head gets 5x encoder LR (1e-4)
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5

# Cross-validation
N_FOLDS = 5
SEED = 42

# Hardware
GPU_DEVICE = 1          # RTX 2080 Ti with 11GB free

# Bloom's taxonomy
BLOOM_LEVELS = {1: "Remember", 2: "Understand", 3: "Apply", 4: "Analyze", 5: "Evaluate", 6: "Create"}
NUM_LEVELS = 6
MIN_LEVEL = 1.0
MAX_LEVEL = 6.0
