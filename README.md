# CogBench: A Benchmark for Evaluating Cognitive-Level Control in LLM Question Generation

CogBench is a benchmark for evaluating whether large language models can generate educational questions at specific cognitive levels according to Bloom's Taxonomy (Anderson & Krathwohl, 2001).

## CogBench v2 (NeurIPS 2026)

The evaluation harness for the NeurIPS 2026 Evaluations & Datasets Track paper (28 deterministic
constraints, standard and adversarial modes, 13-model leaderboard) is in [`v2/`](v2/). The rest of
this repository is the earlier v1 release (question collection, CCS classifier, baseline benchmark).

## Overview

LLMs are increasingly used to generate educational content, but there is no standardized benchmark to evaluate whether they can generate questions at **specific cognitive levels**. CogBench fills this gap by providing:

- **27,099 questions** across 16 academic subjects with Bloom's Taxonomy labels
- **CCS (Cognitive Classification Score) model** — a fine-tuned BERT classifier achieving 82% accuracy on 6-level Bloom's classification
- **Benchmark evaluation framework** — test any LLM's ability to control cognitive level in question generation
- **4 prompt template variants** for robustness analysis

## Bloom's Taxonomy Levels

| Level | Name | Description |
|-------|------|-------------|
| L1 | Remember | Retrieve knowledge from memory |
| L2 | Understand | Construct meaning from information |
| L3 | Apply | Use a procedure in a given situation |
| L4 | Analyze | Break into parts, find relationships |
| L5 | Evaluate | Make judgments based on criteria |
| L6 | Create | Put elements together into a new whole |

## Quick Start

### Installation

```bash
git clone https://github.com/mouryat9/CogBench.git
cd CogBench
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Score a Question

```python
from ccs_model.predict import CCSPredictor

predictor = CCSPredictor(mode="classification")
score = predictor.predict("Compare and contrast mitosis and meiosis", subject="biology")
print(f"Level: {score.level} ({score.name}), Confidence: {score.confidence:.2f}")
# Level: 4 (Analyze), Confidence: 0.95
```

### Run Benchmark Evaluation

```bash
# Generate questions with a local model
python benchmark/generate.py --model qwen2.5:14b --template with_definition

# Score generated questions with CCS
python benchmark/score.py --all

# View results
python benchmark/metrics.py
```

## Dataset

The CogBench dataset contains 27,099 questions from 4 sources:

| Source | Questions | Labels |
|--------|-----------|--------|
| Yahya (2012) | 600 | Human-labeled (6 levels) |
| Mohammed & Omar (2020) | 141 | Human-labeled (6 levels) |
| OpenStax QA | ~9,900 | CCS silver-labeled |
| OpenStax MCQ | ~16,400 | CCS silver-labeled |

**Subjects:** Anthropology, Astronomy, Biology, Business, Chemistry, Computer Science, Economics, History, Mathematics, Nursing, Philosophy, Physics, Political Science, Psychology, Sociology

## CCS Model

The Cognitive Classification Score (CCS) model is a fine-tuned BERT-base-uncased classifier for automatic Bloom's Taxonomy level prediction.

**Phase 1 Results (5-fold CV, n=739):**

| Method | Exact Accuracy | Adjacent (±1) | F1 Macro |
|--------|---------------|----------------|----------|
| Random Baseline | 16.8% | 44.5% | 0.168 |
| Verb Heuristic | 60.6% | 74.4% | 0.613 |
| TF-IDF + SVM | 77.7% | 85.3% | 0.776 |
| LLM Panel (4-vote) | 74.2% | 84.3% | 0.744 |
| **CCS (BERT)** | **82.0%** | **89.6%** | **0.819** |

## Benchmark Results (Preliminary)

Cognitive-level control accuracy when generating questions (with_definition template):

| Model | Level Accuracy | Adjacent Accuracy | CCS-Control |
|-------|---------------|-------------------|-------------|
| phi4:14b | 86.8% | 90.3% | 0.893 |
| gemma3:12b | 81.2% | 88.2% | 0.856 |
| qwen2.5:14b | 75.7% | 84.7% | 0.813 |
| llama3.1:8b | 64.6% | 76.4% | 0.725 |

**Key finding:** All models struggle with L2 (Understand) and L3 (Apply) — they tend to overshoot to higher cognitive levels.

## Project Structure

```
cogbench/
├── benchmark/           # Benchmark evaluation framework
│   ├── generate.py      # Generate questions from LLMs
│   ├── score.py         # Score generated questions with CCS
│   ├── metrics.py       # Compute benchmark metrics
│   ├── prompt_templates.py  # 4 prompt template variants
│   └── config.py        # Models, subjects, parameters
├── ccs_model/           # CCS classifier
│   ├── model.py         # BERT-base + classification head
│   ├── train.py         # Phase 1: 5-fold CV training
│   ├── predict.py       # Inference module
│   ├── evaluate.py      # Evaluation metrics
│   ├── silver_label.py  # Label full dataset with CCS
│   └── baselines/       # Baseline classifiers
├── collection/          # Data collection scripts
├── analysis/            # Inter-annotator agreement analysis
├── data/                # Dataset files
└── requirements.txt
```

## Citation

If you use CogBench in your research, please cite:

```bibtex
@misc{kunuku2026cogbench,
  title={CogBench: A Benchmark for Evaluating Cognitive-Level Control in LLM Question Generation},
  author={Kunuku, Mourya Teja},
  year={2026},
  url={https://github.com/mouryat9/CogBench}
}
```

## License

CC-BY-4.0
