"""Prompt templates for CogBench benchmark evaluation.

4 prompt variants to test LLM cognitive-level control:
  1. name_only       — Just the level name
  2. with_definition — Level name + cognitive verb description
  3. with_exemplar   — Definition + an example question at that level
  4. chain_of_thought — Ask model to reason about the level before generating

Each template takes: level_name, level_number, subject, definition, exemplar
"""

# ─── Bloom's Level Definitions (Anderson & Krathwohl, 2001) ───────────────────

BLOOM_DEFINITIONS = {
    1: {
        "name": "Remember",
        "definition": "retrieve relevant knowledge from long-term memory",
        "verbs": "define, list, recall, identify, name, state, recognize",
        "student_action": "recall facts, terms, basic concepts, or answers",
    },
    2: {
        "name": "Understand",
        "definition": "construct meaning from instructional messages, including oral, written, and graphic communication",
        "verbs": "explain, describe, summarize, interpret, paraphrase, classify",
        "student_action": "demonstrate understanding of facts and ideas by organizing, comparing, translating, interpreting, giving descriptions, and stating main ideas",
    },
    3: {
        "name": "Apply",
        "definition": "carry out or use a procedure in a given situation",
        "verbs": "calculate, solve, demonstrate, implement, use, execute",
        "student_action": "solve problems in new situations by applying acquired knowledge, facts, techniques, and rules in a different way",
    },
    4: {
        "name": "Analyze",
        "definition": "break material into its constituent parts and determine how the parts relate to one another and to an overall structure or purpose",
        "verbs": "compare, contrast, examine, differentiate, categorize, organize",
        "student_action": "examine and break information into parts, identify motives or causes, make inferences, and find evidence to support generalizations",
    },
    5: {
        "name": "Evaluate",
        "definition": "make judgments based on criteria and standards",
        "verbs": "justify, critique, assess, argue, defend, judge, appraise",
        "student_action": "present and defend opinions by making judgments about information, validity of ideas, or quality of work based on a set of criteria",
    },
    6: {
        "name": "Create",
        "definition": "put elements together to form a coherent or functional whole; reorganize elements into a new pattern or structure",
        "verbs": "design, propose, construct, develop, formulate, synthesize",
        "student_action": "compile information together in a different way by combining elements in a new pattern or proposing alternative solutions",
    },
}

# ─── Exemplar Questions (from labeled dataset, curated for clarity) ───────────

EXEMPLAR_QUESTIONS = {
    1: {  # Remember
        "biology": "What is the basic unit of life?",
        "chemistry": "What is the chemical formula for water?",
        "physics": "What is Newton's first law of motion?",
        "mathematics": "What is the formula for the area of a circle?",
        "history": "In what year did World War II end?",
        "psychology": "Who is considered the founder of psychoanalysis?",
        "economics": "What is the definition of GDP?",
        "default": "What is the definition of photosynthesis?",
    },
    2: {  # Understand
        "biology": "Explain how the process of osmosis differs from diffusion.",
        "chemistry": "Describe how ionic bonds form between atoms.",
        "physics": "Explain why objects in free fall experience weightlessness.",
        "mathematics": "Describe the relationship between the derivative and the slope of a tangent line.",
        "history": "Summarize the main causes of the French Revolution.",
        "psychology": "Explain the difference between classical and operant conditioning.",
        "economics": "Describe how supply and demand determine market price.",
        "default": "Explain how photosynthesis converts light energy into chemical energy.",
    },
    3: {  # Apply
        "biology": "Using the Hardy-Weinberg equation, calculate the frequency of heterozygous carriers.",
        "chemistry": "Calculate the molarity of a solution containing 5.85g of NaCl in 500mL of water.",
        "physics": "A 2kg object is dropped from 10m. Calculate its velocity just before hitting the ground.",
        "mathematics": "Solve the equation 2x² + 5x - 3 = 0 using the quadratic formula.",
        "history": "Apply the concept of containment policy to explain U.S. involvement in the Korean War.",
        "psychology": "Apply Maslow's hierarchy of needs to explain a teenager's behavior in a school setting.",
        "economics": "Using the multiplier effect, calculate the total change in GDP from a $100 million increase in government spending with MPC of 0.8.",
        "default": "Calculate the rate of photosynthesis given that a plant produces 12mL of oxygen in 5 minutes.",
    },
    4: {  # Analyze
        "biology": "Compare and contrast the processes of mitosis and meiosis in terms of their outcomes and biological significance.",
        "chemistry": "Examine the relationship between molecular structure and boiling point in the alcohol homologous series.",
        "physics": "Differentiate between the wave and particle models of light, citing evidence for each.",
        "mathematics": "Analyze the relationship between a function's continuity and differentiability.",
        "history": "Compare the political systems of Athens and Sparta and their influence on modern democracy.",
        "psychology": "Examine the similarities and differences between Freud's and Jung's theories of the unconscious.",
        "economics": "Analyze how changes in interest rates affect both consumer spending and business investment differently.",
        "default": "Compare the light-dependent and light-independent reactions of photosynthesis.",
    },
    5: {  # Evaluate
        "biology": "Evaluate the ethical implications of using CRISPR gene editing in human embryos.",
        "chemistry": "Assess whether green chemistry practices can realistically replace traditional industrial chemical processes.",
        "physics": "Critique the argument that nuclear energy is the best solution to climate change.",
        "mathematics": "Judge which numerical method (Newton's vs. bisection) is more appropriate for finding roots of non-smooth functions and justify your answer.",
        "history": "Evaluate whether the Treaty of Versailles was primarily responsible for the rise of Nazism in Germany.",
        "psychology": "Assess the validity of using IQ tests as a measure of intelligence, considering cultural bias.",
        "economics": "Evaluate the argument that minimum wage increases lead to unemployment.",
        "default": "Assess whether artificial photosynthesis is a viable solution to global energy needs.",
    },
    6: {  # Create
        "biology": "Design an experiment to test whether a specific gene is responsible for antibiotic resistance in bacteria.",
        "chemistry": "Propose a novel method for removing microplastics from ocean water using chemical principles.",
        "physics": "Design a thought experiment that demonstrates the principle of equivalence in general relativity.",
        "mathematics": "Formulate a mathematical model that predicts the spread of a disease in a population.",
        "history": "Construct an alternative historical narrative: What might have happened if the Allies lost the Battle of Midway?",
        "psychology": "Design a study to investigate the effect of social media usage on adolescent self-esteem.",
        "economics": "Develop a policy proposal to reduce income inequality while maintaining economic growth.",
        "default": "Design an experiment to determine the optimal conditions for maximizing the rate of photosynthesis.",
    },
}


def get_exemplar(level: int, subject: str) -> str:
    """Get an exemplar question for a given level and subject."""
    level_exemplars = EXEMPLAR_QUESTIONS.get(level, {})
    return level_exemplars.get(subject, level_exemplars.get("default", ""))


# ─── Prompt Templates ─────────────────────────────────────────────────────────

def _article(word: str) -> str:
    """Return 'an' for vowel-starting words, 'a' otherwise."""
    return "an" if word[0].lower() in "aeiou" else "a"


def name_only(level: int, subject: str) -> str:
    """Variant 1: Just the level name. Minimal instruction."""
    level_name = BLOOM_DEFINITIONS[level]["name"]
    art = _article(level_name)
    return (
        f"Generate {art} {level_name}-level question about {subject}.\n\n"
        f"Respond with ONLY the question. No explanation, no preamble."
    )


def with_definition(level: int, subject: str) -> str:
    """Variant 2: Level name + cognitive definition + action verbs."""
    info = BLOOM_DEFINITIONS[level]
    return (
        f"Generate a question about {subject} at Bloom's Taxonomy {info['name']} level.\n\n"
        f"At the {info['name']} level, students should be able to {info['definition']}. "
        f"Typical action verbs: {info['verbs']}.\n\n"
        f"The question should require students to {info['student_action']}.\n\n"
        f"Respond with ONLY the question. No explanation, no preamble."
    )


def with_exemplar(level: int, subject: str) -> str:
    """Variant 3: Definition + an example question at that level."""
    info = BLOOM_DEFINITIONS[level]
    exemplar = get_exemplar(level, subject)
    return (
        f"Generate a question about {subject} at Bloom's Taxonomy {info['name']} level.\n\n"
        f"At the {info['name']} level, students should be able to {info['definition']}. "
        f"Typical action verbs: {info['verbs']}.\n\n"
        f"Here is an example of a {info['name']}-level question:\n"
        f"\"{exemplar}\"\n\n"
        f"Generate a DIFFERENT question at the same cognitive level about {subject}.\n\n"
        f"Respond with ONLY the question. No explanation, no preamble."
    )


def chain_of_thought(level: int, subject: str) -> str:
    """Variant 4: Ask the model to reason about the level, then generate."""
    info = BLOOM_DEFINITIONS[level]
    return (
        f"You are an expert in Bloom's Taxonomy (Anderson & Krathwohl, 2001).\n\n"
        f"Task: Generate a question about {subject} at the {info['name']} level.\n\n"
        f"Step 1: Briefly explain what makes a question {info['name']}-level "
        f"(what cognitive process it requires from students).\n\n"
        f"Step 2: Generate a question about {subject} that specifically targets "
        f"the {info['name']} cognitive level.\n\n"
        f"Format your response as:\n"
        f"REASONING: [your explanation]\n"
        f"QUESTION: [the generated question]"
    )


# ─── Template Registry ────────────────────────────────────────────────────────

TEMPLATES = {
    "name_only": name_only,
    "with_definition": with_definition,
    "with_exemplar": with_exemplar,
    "chain_of_thought": chain_of_thought,
}


def get_prompt(template_name: str, level: int, subject: str) -> str:
    """Get a formatted prompt for a given template, level, and subject."""
    if template_name not in TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}. Choose from: {list(TEMPLATES.keys())}")
    return TEMPLATES[template_name](level, subject)


def list_templates() -> list:
    """Return available template names."""
    return list(TEMPLATES.keys())
