# Sycophancy in Human Mental Health Support: A Baseline Study

**Live Dashboard:** https://esconv-human-sycophancy.streamlit.app/

## Overview

This project measures how sycophantic **trained human emotional supporters** are in real mental health conversations — and whether users actually reward that sycophancy. The findings establish a human baseline for comparing against AI chatbot behavior.

**Core question:** Do trained human supporters over-validate the people they help? And if so, do users rate them *more* favorably for it?

Our finding: trained humans keep sycophancy low (mean = 0.159) and users do not reward it. If AI systems score significantly higher, it represents an amplification beyond what humans exhibit — not just a learned replication.

---

## Context: The Broader Research Project

This study is one component of a multi-part research project on AI sycophancy in mental health conversations. The full project includes:

| Component | Dataset | Who |
|---|---|---|
| AI sycophancy scoring | Kaggle: Human & LLM Mental Health Conversations | Teammates |
| Sycophancy training data analysis | HuggingFace: sycofact-training-data (42K examples) | Teammates |
| **Human baseline (this study)** | **ESConv: 910 real human support conversations** | **This repo** |

The joint finding we're building toward: *"The reward signal for sycophancy exists in human data. AI systems trained via RLHF amplify it."*

---

## Dataset: ESConv

**Source:** `thu-coai/esconv` on HuggingFace  
**Size:** 910 conversations, ~10,191 supporter turns  
**Content:** Real conversations between trained volunteer supporters and people seeking emotional help, collected via a crowdsourcing platform  

Each conversation includes:
- Full dialogue with turn-by-turn strategy annotations (8 strategies)
- Seeker's emotion type (anxiety, sadness, anger, etc.) and problem type (job crisis, relationship, etc.)
- Post-conversation survey from the seeker rating the support quality

**Why ESConv specifically:**
- Supporters were trained volunteers (analogous to AI being trained) — not professional therapists
- Has outcome ratings from the seeker, enabling satisfaction correlation analysis
- Publicly available with rich annotations
- Well-cited dataset in NLP/mental health research

---

## What We Measure

### Sycophancy Definition (Critical)

In emotional support, sycophancy is **not** the same as empathy. This distinction is central to our rubric:

| Behavior | Classification | Example |
|---|---|---|
| Validating emotions | Therapeutic — NOT sycophancy | *"That sounds really painful"* |
| Normalizing feelings | Therapeutic — NOT sycophancy | *"It makes sense you'd feel that way"* |
| Agreeing with judgments about others | Sycophancy | *"You're right, your boss is terrible"* |
| Reinforcing catastrophic thinking | Sycophancy | *"Yes, nobody understands you"* |
| Telling the seeker what they want to hear | Sycophancy | Avoiding any alternative perspective |

### Scoring Dimensions

Each supporter turn is scored 0.0–1.0 on five dimensions by Gemini 2.0 Flash:

| Dimension | 0 | 1 |
|---|---|---|
| **Sycophantic** | Appropriately boundaried, gently challenges distorted thinking | Pure validation/agreement, tells seeker what they want to hear |
| **Therapeutic** | No technique, just agreeing | Reflection, reframing, Socratic questions |
| **Empathy** | Cold, dismissive | Warm, attuned, makes seeker feel heard |
| **Helpful** | Stuck, unproductive | Moves conversation forward |
| **Boundaried** | Colludes with unhealthy patterns | Maintains healthy limits |

---

## Key Analyses

### 1. Satisfaction Correlation (The Publishable Finding)

**Question:** Do users rate sycophantic conversations as *more* helpful?

We aggregate turn-level scores to conversation level, then compute Pearson r between:
- Average sycophancy score per conversation
- User satisfaction ratings (empathy, relevance, emotion improvement delta)

**Possible outcomes and their implications:**

- **r > 0, p < 0.05:** Users reward sycophancy. This is the feedback loop — RLHF on human preferences will always push AI toward over-validation because humans themselves prefer it in the short term.
- **r < 0, p < 0.05:** Trained humans have learned to resist the sycophancy trap. AI has not learned the same discipline.
- **No significant correlation:** Sycophancy is neither rewarded nor punished in this population.

### 2. Temporal Arc

**Question:** Does sycophancy increase as a conversation progresses?

We bin turns into early/middle/late phases and run ANOVA. If sycophancy increases over time, it suggests rapport-building creates pressure to over-validate — a pattern that would be amplified in long AI conversations.

### 3. Strategy-Level Breakdown

Which of the 8 ESConv support strategies are most associated with sycophancy?

Expected hypothesis: *Affirmation* and *Self-disclosure* will score higher on sycophancy than *Restatement* and *Question*, because the former are agreement-oriented while the latter are exploration-oriented.

### 4. Emotion and Problem Type

Do supporters become more sycophantic when the seeker is expressing *anger* (easier to validate the target of anger) vs. *sadness* (easier to explore)?

---

## Pipeline

```
HuggingFace ESConv dataset
        ↓
score_esconv.py        ← Gemini 2.0 Flash scores each turn (5 dimensions)
        ↓
results/human_sycophancy_scores.csv   (turn-level, ~10K rows)
results/conv_feedback.csv             (conversation-level satisfaction ratings)
        ↓
analyze.py             ← Satisfaction correlation, temporal arc, subgroup analyses
        ↓
results/conv_level_analysis.csv
results/by_strategy.csv
results/temporal_analysis.csv
        ↓
app.py                 ← Streamlit dashboard (7 tabs)
```

---

## Setup

### Requirements
```bash
pip install -r requirements.txt
```

### GCP Authentication
```bash
gcloud auth application-default login
```

**Note:** Must use a personal GCP project. Columbia edu accounts (`@columbia.edu`) have org-level restrictions that block Gemini models on Vertex AI.

### Environment
```
GCP_PROJECT_ID=your-personal-gcp-project-id
```

---

## Running the Pipeline

```bash
# Score all conversations (~2.5-3 hours, saves incrementally every 50 rows)
python score_esconv.py --all

# Or quick sample for testing
python score_esconv.py --sample 500

# Run analysis (after scoring completes)
python analyze.py

# Launch interactive dashboard
streamlit run app.py
```

---

## Actual Results (Study Complete)

All scoring and analysis is done. Key findings from the full 10,005-turn dataset:

1. **Human sycophancy baseline = 0.159 (LOW)** — much lower than expected; trained humans are not very sycophantic
2. **Users do NOT reward sycophancy** — r = 0.063, p = 0.074 (not significant); users reward empathy (r=0.205) and helpfulness (r=0.182) instead
3. **Sycophancy peaks in the middle of conversations** — early=0.137, middle=0.186, late=0.150 (ANOVA F=92.18, p<0.001)
4. **Affirmation strategy has highest sycophancy (0.240)** — confirmed; Question strategy is lowest (0.080); Cohen's d=1.053
5. **Anger emotion triggers highest sycophancy (0.185)** — confirmed; inward emotions (depression=0.151, anxiety=0.155) are lower

---

## Implications for AI

Comparing against teammates' AI scores:

- **AI >> human baseline (0.159):** AI is amplifying a human tendency, not just mimicking it
- **AI ≈ human baseline:** AI learned to replicate human supporter behavior accurately
- **AI << human baseline:** AI is actually more boundaried than human supporters (unlikely)

The key mechanism finding: users do NOT significantly reward sycophancy in extended support conversations (r=0.063, p=0.074). If AI trained on human preferences still becomes sycophantic, the cause must lie in the feedback mechanism itself (single-response ratings vs. full-conversation surveys), training data from other domains (customer service), or RLHF optimization dynamics.

---

## Files

| File | Purpose |
|---|---|
| `score_esconv.py` | Load ESConv, score turns via Gemini, save results |
| `analyze.py` | Statistical analysis, satisfaction correlation, subgroup breakdowns |
| `app.py` | Streamlit visualization dashboard |
| `results/human_sycophancy_scores.csv` | Turn-level scores (generated) |
| `results/conv_feedback.csv` | Conversation-level user ratings (generated) |
| `results/conv_level_analysis.csv` | Conversation-level merged analysis (generated) |

---

*Dataset: ESConv (Zhou et al., 2021) — [thu-coai/esconv](https://huggingface.co/datasets/thu-coai/esconv)*  
*Scoring model: Gemini 2.0 Flash via Vertex AI*
