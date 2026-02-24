"""CogBench benchmark configuration."""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "generated_questions")

# ─── Bloom's Levels ───────────────────────────────────────────────────────────

BLOOM_LEVELS = {1: "Remember", 2: "Understand", 3: "Apply",
                4: "Analyze", 5: "Evaluate", 6: "Create"}

# ─── Subjects ─────────────────────────────────────────────────────────────────

# Core subjects (high question count in dataset + diverse domains)
CORE_SUBJECTS = [
    "biology", "chemistry", "physics", "mathematics",
    "psychology", "economics", "history", "computer_science",
]

# Extended subjects (all 16 in the dataset minus 'general')
ALL_SUBJECTS = [
    "anthropology", "astronomy", "biology", "business",
    "chemistry", "computer_science", "economics", "history",
    "mathematics", "nursing", "philosophy", "physics",
    "political_science", "psychology", "sociology",
]

# ─── Models ───────────────────────────────────────────────────────────────────

# Local models (Ollama — free)
LOCAL_MODELS = {
    "qwen2.5:14b": {"backend": "ollama", "name": "Qwen 2.5 14B", "params": "14B"},
    "llama3.1:8b": {"backend": "ollama", "name": "Llama 3.1 8B", "params": "8B"},
    "gemma3:12b": {"backend": "ollama", "name": "Gemma 3 12B", "params": "12B"},
    "phi4:14b": {"backend": "ollama", "name": "Phi-4 14B", "params": "14B"},
    "deepseek-r1:14b": {"backend": "ollama", "name": "DeepSeek-R1 14B", "params": "14B"},
}

# API models (need keys)
API_MODELS = {
    "gpt-4o": {"backend": "openai", "name": "GPT-4o", "params": "~200B"},
    "gpt-4o-mini": {"backend": "openai", "name": "GPT-4o-mini", "params": "~8B"},
    "claude-3-5-sonnet-20241022": {"backend": "anthropic", "name": "Claude 3.5 Sonnet", "params": "~70B"},
    "claude-3-haiku-20240307": {"backend": "anthropic", "name": "Claude 3 Haiku", "params": "~20B"},
    "gemini-1.5-pro": {"backend": "google", "name": "Gemini 1.5 Pro", "params": "~175B"},
}

# ─── Generation Parameters ────────────────────────────────────────────────────

N_SAMPLES = 3               # Generations per (model, level, subject) combination
TEMPERATURE = 0.7           # Diverse but not wild
MAX_TOKENS = 256            # Questions shouldn't be longer than this
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 120        # seconds per generation

# ─── Evaluation Parameters ────────────────────────────────────────────────────

# Prompt templates to evaluate (robustness analysis)
PROMPT_TEMPLATES = ["name_only", "with_definition", "with_exemplar", "chain_of_thought"]
DEFAULT_TEMPLATE = "with_definition"  # Primary template for main results

# CCS model settings for scoring
CCS_MODE = "classification"  # classification or regression
CCS_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "ccs_model", "checkpoints", "ccs_deberta_v1")
