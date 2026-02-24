# CogBench Annotation Guidelines
## Bloom's Taxonomy Question Classification

**Version:** 1.0
**Date:** February 2026
**Project:** CogBench — NeurIPS 2026 Datasets & Benchmarks Track

---

## 1. Task Overview

### What You're Doing

You will read educational questions (from college-level textbooks) and classify each one into **one of six cognitive levels** from the Revised Bloom's Taxonomy (Anderson & Krathwohl, 2001). These levels describe the type of thinking a student must do to answer the question.

### Why This Matters

We are building **CogBench**, a benchmark to evaluate whether AI language models (like ChatGPT, Claude, Gemini) can generate questions at specific cognitive levels. Your labels become the "gold standard" — the ground truth that the AI is measured against. Accurate, consistent labeling is essential for the benchmark to be scientifically valid.

### What You'll Label

- **~300 questions** from 16 subjects (biology, chemistry, physics, mathematics, computer science, psychology, economics, history, and more)
- Questions come from OpenStax textbooks (free, peer-reviewed college textbooks)
- Mix of **multiple-choice (MCQ)** and **open-ended** questions

### For Each Question, You Will:

1. **Assign one Bloom's level** (L1 through L6) — which cognitive process does the question require?
2. **Rate your confidence** (High / Medium / Low)
3. **Optionally write a note** explaining your reasoning, especially for borderline cases

### Important Rules

- Label **independently** — do not discuss specific questions with other annotators during the annotation phase
- Classify based on the **cognitive process the student must perform**, not just the verb in the question
- When in doubt, use "Medium" confidence and write a brief note
- Take breaks every 45-60 minutes to maintain accuracy

---

## 2. Bloom's Taxonomy Quick Reference

The Revised Bloom's Taxonomy (2001) has six levels, from lowest cognitive demand (L1) to highest (L6):

### L1 — Remember
**Definition:** Retrieve relevant knowledge from long-term memory. The student recalls, recognizes, or identifies facts, terms, or concepts without needing to change or process them.

**Key verbs:** define, list, name, recall, identify, state, label, recognize, match, memorize

**Examples:**
- "What is the chemical formula for water?" — Student recalls H2O. *Pure recall.*
- "List the four chambers of the human heart." — Student reproduces memorized anatomy. *Retrieval without transformation.*

**Trap:** "Describe the structure of DNA." — If the answer is just listing components (double helix, nucleotides, sugar-phosphate backbone), this is L1 (recall). But if the student must explain HOW the structure enables function, that's L2.

---

### L2 — Understand
**Definition:** Construct meaning from information. The student explains, summarizes, paraphrases, classifies, or interprets — demonstrating comprehension beyond mere recall.

**Key verbs:** explain, describe, summarize, paraphrase, classify, interpret, discuss, illustrate, give an example

**Examples:**
- "Explain how osmosis works in your own words." — Student must rephrase the concept, showing understanding. *Not just reciting a definition.*
- "Classify these animals as vertebrates or invertebrates." — Student applies understanding of categories. *Sorting based on comprehension.*

**Trap:** "Explain Newton's Third Law" — If the student can copy the textbook wording, this is L1. If they must rephrase it or give a novel example, this is L2. **Context matters** — when in doubt and the question just says "explain" a well-known concept, lean toward L2.

---

### L3 — Apply
**Definition:** Carry out or use a procedure in a given or new situation. The student executes a known method, calculates a result, or demonstrates a technique.

**Key verbs:** calculate, solve, apply, use, demonstrate, implement, compute, determine, show

**Examples:**
- "Calculate the pH of a 0.1M HCl solution." — Student uses the pH formula. *Executing a known procedure.*
- "Use the quadratic formula to solve x² + 5x + 6 = 0." — Student applies the formula to a specific problem. *Procedure application.*

**Trap:** "Apply the concept of supply and demand to explain why gasoline prices rose." — Despite using the word "apply," this requires analysis of a real-world situation, making it L4 (Analyze), not L3. **L3 is for straightforward procedural execution**, not applying concepts to explain complex phenomena.

---

### L4 — Analyze
**Definition:** Break material into constituent parts and determine how parts relate to one another and to an overall structure or purpose. The student compares, contrasts, differentiates, examines cause-effect, or identifies patterns.

**Key verbs:** compare, contrast, differentiate, examine, distinguish, categorize, organize, deconstruct, relate

**Examples:**
- "Compare and contrast mitosis and meiosis." — Student must identify similarities AND differences. *Examining relationships between parts.*
- "What factors contributed to the fall of the Roman Empire?" — Student must identify and relate multiple causal factors. *Breaking a complex event into parts.*

**Trap:** "Compare these two numbers" — Simple comparison might be L1 or L2 depending on context. For L4, the comparison should require examining underlying structure or relationships, not just surface features.

---

### L5 — Evaluate
**Definition:** Make judgments based on criteria and standards. The student critiques, justifies, assesses the quality of something, defends or challenges a position, or judges the effectiveness of an approach.

**Key verbs:** evaluate, justify, critique, assess, argue, defend, judge, support, recommend, determine the validity of

**Examples:**
- "Evaluate whether the author's argument is well-supported by the evidence." — Student must apply criteria (logical consistency, evidence quality) to make a judgment. *Judging quality.*
- "Which of these experimental designs would yield more reliable results, and why?" — Student must assess quality using methodological criteria. *Judgment with justification.*

**Trap:** "Do you think climate change is a problem?" — A simple opinion is NOT L5. Evaluate requires **reasoned judgment based on explicit criteria**. The student must explain WHY, using evidence and standards.

---

### L6 — Create
**Definition:** Put elements together to form a coherent or functional whole. The student designs, constructs, develops, plans, or produces something new or original.

**Key verbs:** design, create, construct, develop, formulate, propose, plan, compose, invent, hypothesize

**Examples:**
- "Design an experiment to test whether fertilizer affects plant growth." — Student must create a novel experimental procedure. *Producing something new.*
- "Propose a policy to reduce urban air pollution." — Student must synthesize knowledge into an original proposal. *Combining ideas into a new whole.*

**Trap:** "Create a list of all the presidents." — Despite using the word "create," this is L1 (recall). **L6 requires genuine synthesis — combining elements into something that didn't exist before.** A list of memorized facts is not creation.

---

## 3. Decision Flowchart

Use this step-by-step process when you're unsure about a question's level:

```
START: Read the question carefully.

Q1: Does the question ask the student to recall, recognize, or
    reproduce a fact, definition, or piece of information AS-IS?
    → YES → L1 REMEMBER
    → NO  → Continue

Q2: Does the question ask the student to explain, summarize,
    paraphrase, interpret, or demonstrate understanding of a
    concept IN THEIR OWN WORDS?
    → YES → L2 UNDERSTAND
    → NO  → Continue

Q3: Does the question ask the student to use a known procedure,
    formula, or method to solve a specific problem?
    (The method is given or obvious — the student just executes it.)
    → YES → L3 APPLY
    → NO  → Continue

Q4: Does the question ask the student to break down information,
    compare/contrast, examine relationships, or identify patterns?
    → YES → L4 ANALYZE
    → NO  → Continue

Q5: Does the question ask the student to make a judgment, critique,
    assess quality, or defend a position using criteria or evidence?
    → YES → L5 EVALUATE
    → NO  → Continue

Q6: Does the question ask the student to design, create, propose,
    or produce something new or original?
    → YES → L6 CREATE
    → NO  → Re-read the question and try again. If still stuck,
            choose your best guess with "Low" confidence and
            write a note explaining why.
```

---

## 4. Boundary Cases — The Tricky Ones

This is the most important section. Research shows that annotators agree easily on the extremes (L1 and L6) but struggle with the middle levels. Here are the four most commonly confused boundaries with worked examples.

### Boundary 1: Remember (L1) vs. Understand (L2)

**The key question:** Can the student answer by simply reproducing information from the textbook, or must they transform it (rephrase, explain, give examples)?

| Question | Level | Why |
|----------|-------|-----|
| "What is photosynthesis?" | L1 | Student can recite the textbook definition verbatim |
| "Explain the process of photosynthesis in your own words." | L2 | "In your own words" requires rephrasing |
| "What are the three branches of government?" | L1 | Listing memorized facts |
| "Why are there three branches of government?" | L2 | Explaining the reasoning requires understanding |
| "Define osmosis." | L1 | Recall of a definition |
| "How is osmosis different from diffusion?" | L2 | Comparing two concepts requires understanding both |

**Rule of thumb:** If the answer would be word-for-word the same for every student → L1. If students would phrase the answer differently → L2.

### Boundary 2: Understand (L2) vs. Analyze (L4)

**The key question:** Is the student explaining a single concept, or examining how parts relate to each other?

| Question | Level | Why |
|----------|-------|-----|
| "Describe supply and demand." | L2 | Explaining a concept |
| "Examine how a shift in supply affects the equilibrium price." | L4 | Analyzing the RELATIONSHIP between two things |
| "Explain the structure of an atom." | L2 | Describing one thing |
| "Compare the atomic structures of carbon and silicon and explain how their differences affect bonding." | L4 | Comparing structures AND linking to consequences |
| "Summarize the causes of World War I." | L2 | Rephrasing known information |
| "Analyze how the alliance system contributed to the escalation of World War I." | L4 | Examining a specific causal mechanism |

**Rule of thumb:** L2 = "Tell me about X." L4 = "How does X relate to Y?" or "Break X into parts and examine them."

### Boundary 3: Apply (L3) vs. Analyze (L4)

**The key question:** Is the student following a given procedure, or deciding which approach to use?

| Question | Level | Why |
|----------|-------|-----|
| "Calculate the derivative of f(x) = x³ + 2x." | L3 | Known procedure (power rule) applied directly |
| "Determine which differentiation technique is appropriate for f(x) = e^x · sin(x), and explain why." | L4 | Student must analyze the function to choose between methods |
| "Balance the equation: Fe + O₂ → Fe₂O₃" | L3 | Standard procedure |
| "Explain why some chemical equations are harder to balance than others." | L4 | Analyzing patterns across multiple examples |
| "Use Ohm's Law to find the current." | L3 | Plug values into V = IR |
| "A circuit has an unexpected reading. Identify which component is malfunctioning and explain how you know." | L4 | Analyzing a situation to diagnose |

**Rule of thumb:** L3 = "Here's the method, now use it." L4 = "Figure out what's going on and why."

### Boundary 4: Evaluate (L5) vs. Create (L6)

**The key question:** Is the student judging something that exists, or producing something new?

| Question | Level | Why |
|----------|-------|-----|
| "Assess whether this experimental design controls for all variables." | L5 | Judging quality of existing work |
| "Design an experiment to test this hypothesis." | L6 | Creating a new experimental procedure |
| "Evaluate the effectiveness of renewable energy policies in reducing carbon emissions." | L5 | Judging existing policies |
| "Propose a new energy policy that addresses both environmental and economic concerns." | L6 | Creating something new |
| "Critique the author's argument about immigration." | L5 | Judging existing argument |
| "Develop your own argument about immigration policy, supporting it with evidence." | L6 | Producing an original argument |

**Rule of thumb:** L5 = "Judge this." L6 = "Make something new."

---

## 5. Special Cases

### MCQ Questions

For multiple-choice questions, classify based on the **cognitive process needed to select the correct answer**, not just what the question stem says.

- "Which of the following is the capital of France? (a) London (b) Paris (c) Berlin" → **L1** (recall)
- "Which of the following best explains why warm air rises? (a) ... (b) ... (c) ..." → **L2** (understand the concept to pick the right explanation)
- "A patient presents with these symptoms. Which diagnosis is most likely?" → **L4** (analyze symptoms to reach a diagnosis)

### Multi-Part Questions

If a question has multiple parts at different levels, classify at the **highest level required**.

- "Define DNA replication (a) and explain why it is important for cell division (b)." → **L2** (part b is L2, which is higher than part a's L1)
- "Calculate the mean (a) and evaluate whether the mean is a good measure of central tendency for this skewed dataset (b)." → **L5** (part b requires evaluation)

### "Describe" Questions

The verb "describe" can appear at multiple levels:
- "Describe the parts of a cell." → **L1** (listing known facts)
- "Describe how a cell responds to changes in osmotic pressure." → **L2** (explaining a process)
- "Describe the similarities and differences between plant and animal cells." → **L4** (comparing = analysis)

### Context-Dependent Questions

Some questions could be different levels depending on what the student was taught. For example, "Solve this integral" could be L3 (Apply) if the technique was taught, or L4 (Analyze) if the student must figure out which technique to use.

**When context is ambiguous:** Assume a standard college course where fundamental techniques have been taught. Classify based on the most likely interpretation.

---

## 6. Using Label Studio

### Interface

When you open Label Studio, you'll see:
1. The **question text** displayed prominently
2. **Subject** and **question type** shown as metadata
3. **Six Bloom's level buttons** (L1-L6) — click one
4. **Three confidence buttons** (High/Medium/Low) — click one
5. **Notes text field** — optional, write your reasoning for unclear cases
6. **Submit button** — saves your annotation and moves to the next question

### Hotkeys (faster annotation)

| Key | Action |
|-----|--------|
| `1` | L1 Remember |
| `2` | L2 Understand |
| `3` | L3 Apply |
| `4` | L4 Analyze |
| `5` | L5 Evaluate |
| `6` | L6 Create |
| `h` | High confidence |
| `m` | Medium confidence |
| `l` | Low confidence |
| `Ctrl+Enter` | Submit and go to next |

### Recommended Pace

- **2-3 minutes per question** is normal
- **25-30 questions per hour** is a sustainable rate
- Take a **5-10 minute break every 45-60 minutes**
- Don't rush — quality over speed
- If you've been staring at a question for >5 minutes, mark it "Low" confidence, write a note, and move on

### Batching

You'll receive questions in batches of ~80. Try to complete one batch per session (2-3 hours). You can stop and resume at any time — your progress is saved automatically.

---

## 7. FAQ

**Q: What if a question doesn't fit any level?**
A: Every question requires some cognitive process, so one of the six levels should apply. If you truly can't decide, pick the closest match, set confidence to "Low," and explain in the notes.

**Q: What if the question is poorly written or unclear?**
A: Classify it based on what you think the question INTENDS to ask. Note "poorly written" in the notes field.

**Q: Should I consider the subject matter when classifying?**
A: No. A recall question is L1 whether it's about biology or history. The cognitive process is what matters, not the subject.

**Q: What if I change my mind about a previous question?**
A: You can go back and modify previous annotations in Label Studio before submitting the batch.

**Q: How many questions will I annotate in total?**
A: ~300 questions over 8 days (150 core questions all annotators share + 150 extended questions assigned to you). Total time commitment is approximately 8-10 hours.

---

## 8. Summary Table

| Level | Name | In One Sentence | Key Verbs |
|-------|------|----------------|-----------|
| L1 | Remember | Recall a fact as-is | define, list, name, recall |
| L2 | Understand | Explain in your own words | explain, summarize, interpret |
| L3 | Apply | Use a known procedure | calculate, solve, apply, use |
| L4 | Analyze | Examine relationships between parts | compare, contrast, examine, differentiate |
| L5 | Evaluate | Judge quality using criteria | evaluate, justify, critique, assess |
| L6 | Create | Produce something new | design, create, propose, develop |

---

*Thank you for contributing to CogBench. Your careful annotations will directly impact the quality of this benchmark and its contribution to AI research.*
