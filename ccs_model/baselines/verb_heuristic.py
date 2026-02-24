"""Verb heuristic baseline — keyword matching on Bloom's action verbs."""

import re
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Action verb lexicon per Bloom's level (from Anderson & Krathwohl 2001)
BLOOM_VERBS = {
    1: [  # Remember
        "define", "list", "name", "recall", "identify", "state", "label",
        "recognize", "match", "memorize", "select", "what is", "who is",
        "when did", "where is", "which of the following",
    ],
    2: [  # Understand
        "explain", "describe", "summarize", "paraphrase", "interpret",
        "classify", "discuss", "illustrate", "give an example", "in your own words",
        "what does", "why does", "how does",
    ],
    3: [  # Apply
        "calculate", "solve", "compute", "demonstrate", "use", "implement",
        "apply", "determine", "find the", "show how", "predict",
        "what would happen if",
    ],
    4: [  # Analyze
        "compare", "contrast", "differentiate", "distinguish", "examine",
        "analyze", "categorize", "break down", "relationship between",
        "cause and effect", "how does .* relate", "what are the differences",
    ],
    5: [  # Evaluate
        "evaluate", "justify", "critique", "assess", "argue", "defend",
        "judge", "support", "which is better", "do you agree",
        "what is the best", "is it valid", "prioritize",
    ],
    6: [  # Create
        "design", "construct", "develop", "formulate", "propose", "create",
        "plan", "compose", "invent", "write a", "build", "generate",
        "how would you design", "devise",
    ],
}


def predict_verb_heuristic(questions):
    """Predict Bloom's levels using verb/keyword matching."""
    predictions = []
    for q in questions:
        text = q["question_text"].lower().strip()
        level = _classify_by_verbs(text)
        predictions.append(level)
    return predictions


def _classify_by_verbs(text):
    """Match text against verb lexicon, return highest matching level."""
    # Check from highest to lowest — higher levels take priority when ambiguous
    for level in range(6, 0, -1):
        for verb in BLOOM_VERBS[level]:
            if re.search(r'\b' + re.escape(verb) if ' ' not in verb else verb, text):
                return level

    # Default: if no verb matches, guess L2 (Understand) as most common
    return 2


if __name__ == "__main__":
    from ccs_model.data_loader import load_labeled_questions
    from ccs_model.evaluate import compute_metrics, print_metrics

    questions = load_labeled_questions()
    preds = predict_verb_heuristic(questions)
    true_levels = [q["bloom_level"] for q in questions]

    metrics = compute_metrics(true_levels, preds)
    print_metrics(metrics, "Verb Heuristic Baseline")

    # Show how many matched each level
    from collections import Counter
    pred_dist = Counter(preds)
    print("Prediction distribution:")
    for level in range(1, 7):
        print(f"  L{level}: {pred_dist.get(level, 0)}")
