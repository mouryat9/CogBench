"""Generate questions from LLMs at specified Bloom's levels.

Usage:
    # Generate with a single local model (quick test)
    python benchmark/generate.py --model qwen2.5:14b --subjects biology physics --quick

    # Full generation with all local models
    python benchmark/generate.py --local-all

    # Specific template
    python benchmark/generate.py --model qwen2.5:14b --template with_exemplar
"""

import os
import sys
import json
import re
import time
import requests
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.config import (
    BLOOM_LEVELS, CORE_SUBJECTS, ALL_SUBJECTS, LOCAL_MODELS,
    N_SAMPLES, TEMPERATURE, MAX_TOKENS, OLLAMA_URL, OLLAMA_TIMEOUT,
    RESULTS_DIR, PROMPT_TEMPLATES, DEFAULT_TEMPLATE,
)
from benchmark.prompt_templates import get_prompt, BLOOM_DEFINITIONS


def generate_ollama(model: str, prompt: str, temperature: float = TEMPERATURE,
                    max_tokens: int = MAX_TOKENS) -> dict:
    """Generate a response from an Ollama model. Returns {text, latency_ms, error}."""
    t0 = time.time()
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        text = resp.json()["response"].strip()
        latency = (time.time() - t0) * 1000
        return {"text": text, "latency_ms": latency, "error": None}
    except Exception as e:
        latency = (time.time() - t0) * 1000
        return {"text": None, "latency_ms": latency, "error": str(e)}


def extract_question(text: str, template_name: str) -> str:
    """Extract the generated question from model output.

    For chain_of_thought template, extract after 'QUESTION:' marker.
    For other templates, take the whole response (model was told to only output the question).
    """
    if text is None:
        return None

    if template_name == "chain_of_thought":
        # Look for QUESTION: marker
        match = re.search(r'QUESTION:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
        if match:
            question = match.group(1).strip()
            # Take first sentence/paragraph (some models add explanation after)
            question = question.split('\n')[0].strip()
            return question

    # For other templates, clean up common artifacts
    cleaned = text.strip()
    # Remove quotes if the model wrapped its answer
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1]
    if cleaned.startswith("'") and cleaned.endswith("'"):
        cleaned = cleaned[1:-1]
    # Remove "Question:" prefix if model added one
    cleaned = re.sub(r'^(Question|Q)\s*:\s*', '', cleaned, flags=re.IGNORECASE)
    # Take first question only (some models generate multiple)
    lines = [l.strip() for l in cleaned.split('\n') if l.strip()]
    if lines:
        cleaned = lines[0]
        # Remove numbering like "1." or "1)"
        cleaned = re.sub(r'^\d+[\.\)]\s*', '', cleaned)

    return cleaned if cleaned else None


def unload_ollama_model(model: str):
    """Unload model from GPU memory."""
    try:
        requests.post(OLLAMA_URL, json={
            "model": model, "prompt": "", "keep_alive": 0,
        }, timeout=10)
    except Exception:
        pass


def generate_for_model(model: str, template_name: str = DEFAULT_TEMPLATE,
                       subjects: list = None, n_samples: int = N_SAMPLES,
                       output_dir: str = None):
    """Generate questions for all (level, subject) combos with one model.

    Saves results incrementally to a JSON file.
    Returns list of generation records.
    """
    subjects = subjects or CORE_SUBJECTS
    output_dir = output_dir or RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    model_safe = model.replace(":", "_").replace(".", "_")
    save_path = os.path.join(output_dir, f"gen_{model_safe}_{template_name}.json")

    # Load existing results for resumability
    existing = {}
    if os.path.exists(save_path):
        with open(save_path) as f:
            data = json.load(f)
            for r in data.get("generations", []):
                key = f"{r['level']}_{r['subject']}_{r['sample_idx']}"
                existing[key] = r
        print(f"  Resuming from {len(existing)} existing generations")

    total = len(BLOOM_LEVELS) * len(subjects) * n_samples
    generations = list(existing.values())
    done = len(existing)
    n_errors = 0
    t_start = time.time()

    print(f"\n  Generating: {model} × {template_name}")
    print(f"  {len(BLOOM_LEVELS)} levels × {len(subjects)} subjects × {n_samples} samples = {total} prompts")

    # Warm up model
    if done == 0:
        print(f"  Warming up {model}...")
        generate_ollama(model, "Hello", temperature=0)

    for level in sorted(BLOOM_LEVELS.keys()):
        for subject in subjects:
            for sample_idx in range(n_samples):
                key = f"{level}_{subject}_{sample_idx}"
                if key in existing:
                    continue

                prompt = get_prompt(template_name, level, subject)
                result = generate_ollama(model, prompt)
                question = extract_question(result["text"], template_name)

                record = {
                    "model": model,
                    "template": template_name,
                    "level": level,
                    "level_name": BLOOM_LEVELS[level],
                    "subject": subject,
                    "sample_idx": sample_idx,
                    "prompt": prompt,
                    "raw_response": result["text"],
                    "generated_question": question,
                    "latency_ms": result["latency_ms"],
                    "error": result["error"],
                    "timestamp": datetime.now().isoformat(),
                }
                generations.append(record)

                if result["error"]:
                    n_errors += 1

                done += 1

                # Progress every 20 generations
                if done % 20 == 0 or done == total:
                    elapsed = time.time() - t_start
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    print(f"  [{model}] {done}/{total} "
                          f"({done/total*100:.0f}%) | "
                          f"{rate:.1f} gen/s | ETA: {eta:.0f}s | "
                          f"Errors: {n_errors}")

                    # Save incrementally
                    _save_generations(save_path, model, template_name, generations)

    # Final save
    _save_generations(save_path, model, template_name, generations)

    elapsed = time.time() - t_start
    print(f"  [{model}] Done: {total} generations in {elapsed:.0f}s, "
          f"{n_errors} errors")
    print(f"  Saved to: {save_path}")

    # Unload model
    print(f"  Unloading {model}...")
    unload_ollama_model(model)
    time.sleep(2)

    return generations


def _save_generations(path, model, template, generations):
    """Save generation results to JSON."""
    with open(path, "w") as f:
        json.dump({
            "model": model,
            "template": template,
            "n_generations": len(generations),
            "timestamp": datetime.now().isoformat(),
            "generations": generations,
        }, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CogBench Question Generation")
    parser.add_argument("--model", type=str, default="qwen2.5:14b",
                        help="Model name (Ollama)")
    parser.add_argument("--template", type=str, default=DEFAULT_TEMPLATE,
                        choices=PROMPT_TEMPLATES,
                        help="Prompt template variant")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Subjects to generate for (default: core 8)")
    parser.add_argument("--all-subjects", action="store_true",
                        help="Use all 15 subjects instead of core 8")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES,
                        help="Samples per (level, subject)")
    parser.add_argument("--local-all", action="store_true",
                        help="Run all local Ollama models sequentially")
    parser.add_argument("--all-templates", action="store_true",
                        help="Run all 4 prompt templates")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 2 subjects × 1 sample")
    args = parser.parse_args()

    subjects = args.subjects
    if args.all_subjects:
        subjects = ALL_SUBJECTS
    elif subjects is None:
        subjects = CORE_SUBJECTS

    n_samples = args.n_samples
    if args.quick:
        subjects = subjects[:2]
        n_samples = 1

    templates = PROMPT_TEMPLATES if args.all_templates else [args.template]
    models = list(LOCAL_MODELS.keys()) if args.local_all else [args.model]

    for model in models:
        for template in templates:
            generate_for_model(
                model=model,
                template_name=template,
                subjects=subjects,
                n_samples=n_samples,
            )
