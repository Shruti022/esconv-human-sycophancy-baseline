# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

This is a completed research project by a Columbia University student (ss7592@columbia.edu) for a Data Science and Policy course. The goal was to establish a **human sycophancy baseline** in real emotional support conversations, and to test whether users actually reward sycophancy.

The project is one part of a larger team effort on AI sycophancy in mental health:

| Component | Dataset | Who |
|---|---|---|
| AI sycophancy scoring | Kaggle: Human & LLM Mental Health Conversations | Teammates |
| Sycophancy training data analysis | HuggingFace: sycofact-training-data | Teammates |
| **Human baseline (this repo)** | **ESConv: 910 real human support conversations** | **This project** |

The comparison script `compare_ai_vs_human.py` is ready to run once teammates share their AI scores.

---

## Current State (Completed)

**Pipeline is fully complete.** All scoring and analysis has been run. Results are saved in `results/`.

- 10,005 turns scored (10,191 extracted, 186 parse errors = 1.8% excluded)
- 910 conversations, 800 with seeker satisfaction surveys
- All analysis scripts have been run; output CSVs exist
- Streamlit dashboard (`app.py`) is working
- `REPORT.md` is the final academic report (submitted quality)
- `FINDINGS.md` is a detailed bullet-point findings reference

---

## Key Verified Findings (All Numbers Checked Against Raw Data)

### Baseline Stats (from `results/human_sycophancy_scores.csv`, n=10,005)
| Dimension | Mean | SD | Median |
|---|---|---|---|
| sycophantic | 0.159 | 0.166 | 0.100 |
| therapeutic | 0.311 | 0.207 | -- |
| empathy | 0.602 | 0.204 | -- |
| helpful | 0.407 | 0.458 | -- |
| boundaried | 0.782 | 0.192 | -- |

Distribution: 71.5% of responses < 0.2, 6.0% > 0.5, 0.7% > 0.7

### Satisfaction Correlations (from `results/conv_level_analysis.csv`, n=800)
| Dimension | vs. Overall Satisfaction | vs. Empathy Rating | vs. Relevance | vs. Emotion Delta |
|---|---|---|---|---|
| sycophantic | r=0.063, p=0.074 n.s. | r=0.077, p=0.030 * | r=0.036 n.s. | r=0.027 n.s. |
| therapeutic | r=0.167, p<0.001 *** | r=0.124 *** | r=0.184 *** | r=0.050 n.s. |
| empathy | r=0.205, p<0.001 *** | r=0.187 *** | r=0.186 *** | r=0.027 n.s. |
| helpful | r=0.182, p<0.001 *** | r=0.150 *** | r=0.185 *** | r=0.068 n.s. |
| boundaried | r=0.017 n.s. | r=-0.000 n.s. | r=0.033 n.s. | r=-0.028 n.s. |

**THE KEY FINDING: Users do NOT significantly reward sycophancy (r=0.063, p=0.074).**
This contradicts the common assumption that RLHF rewards sycophancy in human feedback.

Multivariate regression (all 5 dims predicting satisfaction): R²=0.048, CV R²=0.030
- Standardized beta for sycophantic = +0.083 (slight positive when controlling for others)

### Temporal Arc (ANOVA F=92.18, p<0.001)
| Phase | Sycophancy | Boundaried | Therapeutic | N |
|---|---|---|---|---|
| early (0-33%) | 0.137 | 0.809 | 0.336 | 4146 |
| middle (33-67%) | 0.186 | 0.747 | 0.326 | 4079 |
| late (67-100%) | 0.150 | 0.802 | 0.217 | 1780 |

### Strategy Breakdown (ANOVA F=161.01, p<0.001)
| Strategy | Sycophancy | Boundaried | Empathy | N |
|---|---|---|---|---|
| Affirmation and Reassurance | 0.240 | 0.695 | 0.628 | 1657 |
| Self-disclosure | 0.210 | 0.730 | 0.651 | 964 |
| Reflection of feelings | 0.204 | 0.737 | 0.683 | 854 |
| Providing Suggestions | 0.151 | 0.767 | 0.582 | 1571 |
| Restatement or Paraphrasing | 0.148 | 0.800 | 0.676 | 707 |
| Information | 0.146 | 0.772 | 0.526 | 674 |
| Others | 0.130 | 0.834 | 0.510 | 1811 |
| Question | 0.080 | 0.874 | 0.625 | 1767 |

Affirmation vs Question: t=31.19, p<0.001, Cohen's d=1.053 (large effect)
Note: Empathy is essentially equal (0.628 vs 0.625) -- sycophancy and empathy are separable.

### Emotion Breakdown
| Emotion | Sycophancy | N |
|---|---|---|
| pain | 0.289 | 9 (small) |
| jealousy | 0.193 | 14 (small) |
| anger | 0.185 | 827 |
| disgust | 0.185 | 326 |
| nervousness | 0.167 | 63 |
| sadness | 0.162 | 2413 |
| anxiety | 0.155 | 2680 |
| shame | 0.154 | 256 |
| fear | 0.154 | 764 |
| depression | 0.151 | 2644 |
| guilt | 0.133 | 9 (small) |

### Internal Correlations (turn-level, n=10,005)
| Pair | r |
|---|---|
| sycophantic x boundaried | -0.784 *** |
| therapeutic x empathy | 0.604 *** |
| sycophantic x empathy | 0.135 *** |
| sycophantic x therapeutic | -0.054 *** |
| sycophantic x helpful | -0.000 n.s. |

---

## Run Commands

```bash
# Authenticate with GCP (one-time setup)
gcloud auth application-default login

# Score all turns (~2.5-3 hours, saves every 50 rows)
python score_esconv.py --all

# Score a small sample for testing
python score_esconv.py --sample 500

# Run all analyses (fast, outputs to results/)
python analyze.py

# Launch interactive dashboard
streamlit run app.py

# Compare against teammates' AI scores (run once they share data)
python compare_ai_vs_human.py --ai their_scores.csv --col sycophantic
```

---

## GCP Setup (Required for Scoring)

- Must use a **personal** GCP project (Columbia edu accounts are blocked from Gemini on Vertex AI due to org policies)
- Current project: `second-conquest-487000-r0` (stored in `.env`)
- Authentication: `gcloud auth application-default login`
- Uses `google-genai` SDK with `vertexai=True` backend (NOT the deprecated `vertexai` SDK, NOT AI Studio API key)
- AI Studio API key quota = 0 for Columbia edu accounts

`.env` format:
```
GCP_PROJECT_ID=second-conquest-487000-r0
GOOGLE_API_KEY=...  # unused for scoring; kept for reference
```

---

## Architecture

**Pipeline:** `score_esconv.py` → `analyze.py` → `app.py`

### score_esconv.py
- Loads ESConv from HuggingFace (`thu-coai/esconv`) -- each row is a JSON string in `text` field, parsed with `json.loads`
- Turn fields: `turn['text']` (NOT `content`), `turn.get('strategy', 'unknown')` (top-level, NOT nested in annotation)
- Feedback: `conv.get('survey_score', {}).get('seeker', {})` gives `empathy`, `relevance`, `initial_emotion_intensity`, `final_emotion_intensity`
- Scores via `gemini-2.0-flash` on Vertex AI; 5 dimensions per turn
- Saves `results/human_sycophancy_scores.csv` (turn-level) and `results/conv_feedback.csv`
- JSON parsing uses regex fallback `re.search(r'\{[^{}]*\}', text, re.DOTALL)` for malformed Gemini output

### analyze.py
1. `satisfaction_correlation()` -- aggregates turn scores to conversation level, merges with feedback, Pearson r
2. `temporal_analysis()` -- bins by `turn_position` (pre-normalized 0-1 per conversation), ANOVA
3. `by_strategy()`, `by_emotion()`, `by_problem()` -- subgroup ANOVA breakdowns
Outputs: `conv_level_analysis.csv`, `by_strategy.csv`, `by_emotion.csv`, `by_problem.csv`, `temporal_analysis.csv`, `summary.txt`

### app.py
Streamlit dashboard, 7 tabs: Distributions, Satisfaction Correlation, Temporal Arc, By Strategy, By Emotion, Correlations, Examples.

### compare_ai_vs_human.py
Ready-to-run comparison script. Uses `HUMAN_BASELINE` dict with all verified stats. Runs Welch t-test and Cohen's d vs AI scores. Run when teammates share their CSV.

---

## Critical Rubric Distinction

Sycophancy in mental health support is NOT the same as empathy:
- "That sounds really painful" -- therapeutic validation, NOT sycophancy
- "You're right, your boss is terrible" -- sycophancy (agreeing with judgment without exploration)

This distinction is load-bearing. Conflating them produces noise in scores.

---

## Dataset Notes

- ESConv (Zhou et al., 2021): `thu-coai/esconv` on HuggingFace
- 910 conversations, ~10,191 supporter turn pairs extracted (train split only)
- 800/910 conversations have `survey_score.seeker` feedback
- Supporters are trained volunteers (not professional therapists), analogous to AI trained on guidelines
- 8 support strategies: Question, Restatement or Paraphrasing, Reflection of feelings, Self-disclosure, Affirmation and Reassurance, Providing Suggestions, Information, Others
- Seeker emotions: anxiety (n=2680), depression (n=2644), sadness (n=2413), fear (n=764), anger (n=827), disgust (n=326), shame (n=256), nervousness (n=63), plus small-sample categories

---

## Key Output Files

| File | Contents |
|---|---|
| `results/human_sycophancy_scores.csv` | Turn-level scores: 10,005 rows, columns: sycophantic, therapeutic, empathy, helpful, boundaried, conv_id, turn_id, turn_position (0-1), strategy, emotion_type, etc. |
| `results/conv_feedback.csv` | Conversation-level seeker survey: feedback_empathy, feedback_relevance, feedback_emotion_delta, etc. |
| `results/conv_level_analysis.csv` | Merged: mean scores per conversation + feedback, n=800 |
| `results/by_strategy.csv` | Strategy-level aggregates |
| `results/by_emotion.csv` | Emotion-level aggregates |
| `results/temporal_analysis.csv` | Phase-level aggregates (early/middle/late) |
| `results/summary.txt` | Quick baseline summary |
| `REPORT.md` | Full academic report (final, submitted quality) |
| `FINDINGS.md` | Detailed bullet-point findings with all tables |

---

## Known Issues

- ~1.8% of rows fail JSON parsing (Gemini returns malformed JSON) -- skipped silently
- Columbia edu GCP projects block Gemini models on Vertex AI -- use personal project
- `turn_position` column is pre-normalized (0 to 1.0) within each conversation; use this for temporal analysis, not `turn_position_raw`
