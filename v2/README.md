# CogBench v2: evaluation harness for the NeurIPS 2026 paper

This folder is the evaluation harness for **"CogBench: Measuring How Bloom-Level Adherence in LLM
Question Generation Depends on Verb Cues"** (Mourya Teja Kunuku, Kennesaw State University;
NeurIPS 2026 Evaluations & Datasets Track). During double-blind review the same code was shared
anonymously at `anonymous.4open.science/r/cogbench-422B`; that link has expired and this folder
replaces it.

- **Passages (CC-BY 4.0, Croissant metadata):** https://huggingface.co/datasets/cogbench-anonymous/cogbench-passages
- **Leaderboard:** https://huggingface.co/spaces/cogbench-anonymous/cogbench-leaderboard
- **License:** code Apache-2.0 (see `pyproject.toml`); passages CC-BY 4.0 (OpenStax).
- **Contact:** mkunuku@students.kennesaw.edu

**What this snapshot is.** It is the anonymized review snapshot (`cogbench` 1.1.0, February 2026):
the 28 constraints, standard and adversarial generation, evaluation, the Ollama, OpenAI, Google and
Together backends, the run scripts, and the constraint tests. It does **not** yet include the
Anthropic (Claude) backend, a model-table entry for `gpt-4o-mini`, or the passage-clustered bootstrap (n = 2000) used for the paper's final
numbers; this snapshot's bootstrap is unclustered with n = 1000. These will be added to this folder.

**Prompt fix.** In adversarial mode, some level descriptions and hints name a verb that the same
prompt forbids. `build_adversarial_prompt` now adds one sentence asking the model to express such
requirements in other words. The paper's results used the original prompts; pass
`paper_prompts=True` to reproduce them exactly.

`human_study_3rater/` holds the protocol and scripts of a three-rater human study that is still in
progress; its results are not reported in the camera-ready paper (see its README).

---

# CogBench

**Verifiable Cognitive Constraint Benchmark for LLM Question Generation**

CogBench evaluates whether large language models can generate educationally appropriate questions at specific Bloom's Taxonomy cognitive levels while satisfying 28 deterministic constraints.

## Features

- **6 Bloom's Taxonomy Levels**: Remember, Understand, Apply, Analyze, Evaluate, Create
- **8 Academic Subjects**: Biology, Chemistry, Physics, Mathematics, Psychology, Economics, History, Computer Science
- **28 Deterministic Constraints**: No LLM-as-judge — every constraint is verifiable and reproducible
- **Adversarial Mode**: Tests robustness by using mismatched cognitive-level vocabulary
- **120 Passages**: From OpenStax open-access textbooks

## Installation

```bash
pip install cogbench
```

For NLP features (key concept extraction with spaCy):
```bash
pip install cogbench[nlp]
python -m spacy download en_core_web_sm
```

## Quick Start

```bash
# Show benchmark configuration
cogbench info

# Run benchmark for a single model (requires Ollama)
cogbench run --model qwen2.5:14b --mode standard

# Run all local models in both modes
cogbench run --local-all --mode both

# Re-evaluate existing results with updated constraints
cogbench evaluate

# Generate leaderboard data
cogbench leaderboard
```

## CLI Commands

| Command | Description |
|---|---|
| `cogbench info` | Show benchmark config, models, and passage stats |
| `cogbench run` | Generate questions and evaluate models |
| `cogbench evaluate` | Re-evaluate existing generation files |
| `cogbench leaderboard` | Populate leaderboard data.json from results |
| `cogbench scrape` | Scrape passages from OpenStax textbooks |
| `cogbench submit` | Submit results to the CogBench leaderboard via GitHub |

## Submitting Results

After running the benchmark, submit your results to the public leaderboard:

```bash
# Submit all completed models
cogbench submit --name "Your Name"

# Submit a specific model
cogbench submit --model qwen2.5:14b --name "Your Name"
```

This creates a GitHub issue with your results. The maintainers will review and add them to the leaderboard. Requires the [GitHub CLI](https://cli.github.com) (`gh`).

## Python API

```python
from cogbenchv2.passages.processor import load_all_passages
from cogbenchv2.evaluation.evaluate import evaluate_generations
from cogbenchv2.evaluation.metrics import compute_metrics

# Load the 120 bundled passages
passages = load_all_passages()

# Evaluate a generation file
evaluations = evaluate_generations("gen_qwen2_5_14b_standard.json", passages)

# Compute metrics
metrics = compute_metrics(evaluations)
print(f"Prompt-level strict: {metrics['prompt_level_strict']['rate']:.1%}")
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `COGBENCH_RESULTS_DIR` | `./data/results` | Where benchmark results are saved |
| `OPENAI_API_KEY` | — | For OpenAI API models |
| `GOOGLE_API_KEY` | — | For Google Gemini models |
| `TOGETHER_API_KEY` | — | For Together.ai models |

## License

Apache 2.0
