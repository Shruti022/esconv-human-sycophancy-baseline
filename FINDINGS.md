# Findings: Human Sycophancy Baseline in Mental Health Support
### ESConv Dataset Analysis — 10,005 Scored Responses, 910 Conversations

---

## What We Did

We took 910 real conversations between trained human volunteers and people seeking emotional support (the ESConv dataset). We scored every single supporter response — 10,005 turns total — on five dimensions using Gemini 2.0 Flash via Vertex AI:

1. **Sycophantic** (0 = appropriately boundaried, 1 = over-validating)
2. **Therapeutic** (0 = no technique, 1 = excellent technique)
3. **Empathy** (0 = cold, 1 = warm and attuned)
4. **Helpful** (0 = unhelpful, 1 = moves conversation forward)
5. **Boundaried** (0 = colludes with unhealthy patterns, 1 = maintains healthy limits)

We then correlated these scores against the seeker's post-conversation satisfaction ratings to answer the core question: **do users reward sycophancy?**

---

## Finding 1: Human Supporters Are NOT Very Sycophantic

**Human sycophancy baseline = 0.159 (LOW)**

| Dimension | Mean | Std | Interpretation |
|---|---|---|---|
| Sycophantic | 0.159 | 0.166 | LOW |
| Therapeutic | 0.311 | 0.207 | MODERATE |
| Empathy | 0.602 | 0.204 | HIGH |
| Helpful | 0.407 | 0.458 | MODERATE |
| Boundaried | 0.782 | 0.192 | HIGH |

The sycophancy distribution is heavily right-skewed — the vast majority of responses cluster at 0–0.1, with very few exceeding 0.5. This means most human supporters most of the time are NOT simply agreeing with everything the seeker says.

**What this means for AI comparison:**

This is the baseline. If your teammates find that AI chatbots score significantly higher (e.g., 0.35–0.50), it means AI is 2–3x more sycophantic than trained human supporters. That's not a small difference — it represents a qualitative shift in behavior, not just a marginal one.

The human profile (high empathy + high boundaries + low sycophancy) is exactly what good emotional support looks like theoretically. Trained humans have largely learned to be warm without being agreeable. The question is whether AI has learned the same distinction.

---

## Finding 2: Users Do NOT Reward Sycophancy (The Key Finding)

**Sycophancy × User Satisfaction: r = 0.063, p = 0.074 (not significant)**

We aggregated turn-level scores to conversation level and merged with the seeker's post-conversation survey ratings. The results:

| Score Dimension | vs. Overall Satisfaction | vs. Empathy Rating | vs. Relevance Rating | vs. Emotion Improvement |
|---|---|---|---|---|
| **Sycophantic** | **0.063 (n.s.)** | **0.077 \*** | **0.036 (n.s.)** | **0.027 (n.s.)** |
| Therapeutic | 0.167 \*\*\* | 0.124 \*\*\* | 0.184 \*\*\* | 0.050 (n.s.) |
| Empathy | 0.205 \*\*\* | 0.187 \*\*\* | 0.186 \*\*\* | 0.027 (n.s.) |
| Helpful | 0.182 \*\*\* | 0.150 \*\*\* | 0.185 \*\*\* | 0.068 (n.s.) |
| Boundaried | 0.017 (n.s.) | -0.000 (n.s.) | 0.033 (n.s.) | -0.028 (n.s.) |

*\* p<0.05, \*\*\* p<0.001*

**What users actually reward:** Empathy (r=0.205), helpfulness (r=0.182), and therapeutic technique (r=0.167) — all strongly and significantly predict higher satisfaction ratings.

**What users don't reward:** Sycophancy (r=0.063, not significant). Being agreeable and over-validating does not make users rate the conversation higher.

**What nobody rewards:** Maintaining good boundaries (r=0.017, not significant). Users are essentially blind to whether the supporter challenged their thinking or not.

**The nuanced interpretation:**

This is NOT a null result — it's an important positive finding. Trained human supporters have learned that sycophancy doesn't earn better ratings. They've learned to distinguish between genuine empathy (which users reward) and empty validation (which users don't particularly care about either way).

This raises a critical question for AI: **if human feedback doesn't reward sycophancy, why do AI systems trained on human feedback become sycophantic?**

Possible answers:
1. AI training uses different feedback mechanisms (thumbs up/down, preference rankings) that capture different signals than explicit satisfaction surveys
2. Short AI interactions (single responses) may be rated differently than full conversations
3. AI may be trained on data where sycophancy IS rewarded (social media, customer service) not therapeutic support data
4. The slight positive correlation (r=0.077*) for perceived empathy suggests users partially conflate sycophancy with warmth

---

## Finding 3: Sycophancy Peaks in the Middle of Conversations

**ANOVA: F=92.18, p<0.0001 — significant difference across phases**

| Phase | Avg Sycophancy | Avg Therapeutic | Avg Boundaried |
|---|---|---|---|
| Early (turns 0–33%) | 0.137 | 0.336 | 0.809 |
| **Middle (turns 33–67%)** | **0.186** | **0.326** | **0.747** |
| Late (turns 67–100%) | 0.150 | 0.217 | 0.802 |

Sycophancy follows an **inverted-U arc** across the conversation. Supporters start appropriately boundaried, become more agreeable in the middle as rapport builds, then pull back toward the end.

**Why this happens:**

In the early phase, the supporter is still gathering information — asking questions, clarifying the situation. This is structurally less sycophantic (you can't agree with someone you're still trying to understand).

In the middle phase, rapport has been established. The supporter knows the seeker's perspective and faces social pressure to validate it. The boundaried score drops from 0.809 to 0.747 in this phase — the lowest point in the conversation.

In the late phase, the conversation shifts toward practical support (suggestions, next steps, information). Action-oriented responses are less about validation and more about direction, which naturally reduces sycophancy.

**Implication for AI:** Long AI conversations may be particularly prone to mid-conversation sycophancy drift — the longer the context, the more "rapport" has been established, and the more pressure there is to stay consistent with the seeker's framing rather than challenge it.

---

## Finding 4: Strategy Choice Determines Sycophancy Level

**ANOVA: F=161.01, p<0.0001 — largest effect in the study**

| Strategy | Avg Sycophancy | Avg Therapeutic | Avg Empathy | Avg Boundaried |
|---|---|---|---|---|
| Affirmation and Reassurance | **0.240** | 0.284 | 0.628 | 0.695 |
| Self-disclosure | 0.210 | 0.312 | 0.651 | 0.730 |
| Reflection of Feelings | 0.204 | 0.319 | 0.683 | 0.737 |
| Providing Suggestions | 0.151 | 0.401 | 0.582 | 0.767 |
| Restatement or Paraphrasing | 0.148 | 0.364 | 0.676 | 0.800 |
| Information | 0.146 | 0.319 | 0.526 | 0.772 |
| Others | 0.130 | 0.166 | 0.510 | 0.834 |
| **Question** | **0.080** | 0.375 | 0.625 | **0.874** |

**The headline:** Questioning-based strategies show 3x lower sycophancy than affirmation-based strategies (0.080 vs 0.240).

**Why the top three are more sycophantic:**
- **Affirmation and Reassurance** — by design, this strategy validates the seeker's feelings and strengths. It's the most agreement-oriented strategy in the toolkit.
- **Self-disclosure** — sharing your own experience creates social pressure to agree ("I've felt that way too") — implicitly validating the seeker's framing
- **Reflection of Feelings** — mirroring the seeker's emotions back to them, while valuable, can slide into pure validation if not combined with gentle exploration

**Why Question is lowest:**
Asking questions is structurally incompatible with sycophancy. You cannot simultaneously ask "why do you think that?" and agree with everything the person says. Questions invite the seeker to examine their own thinking rather than validating it.

**The critical insight from comparing Affirmation and Question:**
- Affirmation empathy = 0.628, Question empathy = 0.625 — **essentially identical**
- Affirmation sycophancy = 0.240, Question sycophancy = 0.080 — **3x different**
- Affirmation boundaried = 0.695, Question boundaried = 0.874 — **dramatically different**

This proves that **empathy and sycophancy are separable**. A supporter can be equally warm and attuned using questions vs. affirmations — but one maintains boundaries and the other doesn't. The choice of strategy, not the level of warmth, determines sycophancy.

**Implication for AI:** AI systems that default to affirmation-heavy responses ("That sounds really hard. You're doing the right thing by reaching out. Your feelings are completely valid.") are not being more empathetic — they're just being more sycophantic. The empathy level is the same. The boundary level is not.

---

## Finding 5: Outward-Directed Emotions Trigger More Sycophancy

| Emotion | Avg Sycophancy | N |
|---|---|---|
| Pain | 0.289 | 9 (small sample) |
| Jealousy | 0.193 | 14 (small sample) |
| **Anger** | **0.185** | **827** |
| Disgust | 0.185 | 326 |
| Nervousness | 0.167 | 63 |
| Sadness | 0.162 | 2,413 |
| Anxiety | 0.155 | 2,680 |
| Shame | 0.154 | 256 |
| Fear | 0.154 | 764 |
| Depression | 0.151 | 2,644 |
| Guilt | 0.133 | 9 (small sample) |

**The pattern:** Emotions directed outward (anger, disgust, jealousy) trigger more sycophancy than emotions directed inward (depression, anxiety, shame, guilt).

**Why this makes sense:**

When a seeker is angry at their boss, partner, or a situation, it's socially natural to agree — *"yes, that was unfair"* or *"you're right to be upset."* The supporter can validate the seeker's narrative by agreeing with their judgment of the external target.

When a seeker is depressed or ashamed, the supporter can't really "agree" with the emotion itself. They're forced into responses that acknowledge the feeling without validating a narrative — which is structurally less sycophantic.

**Focus on anger (n=827, largest outward-directed sample):**
Anger is the emotion most prone to triggering sycophancy in supporters. This has implications for AI chatbots deployed in customer service or mental health contexts — when users express anger, AI is most likely to agree with and validate the user's framing of events, even if that framing is distorted.

---

## Finding 6: Sycophancy and Boundaries Are Almost Mirror Opposites

**Sycophantic × Boundaried: r = -0.784, p<0.0001**

The strongest relationship in the entire dataset. These two dimensions are nearly inverse of each other.

| Relationship | r | Interpretation |
|---|---|---|
| Sycophantic × Boundaried | **-0.784** | Strong inverse — sycophancy = loss of boundaries |
| Therapeutic × Empathy | 0.604 | Strong positive — technique and warmth go together |
| Sycophantic × Therapeutic | -0.054 | Near zero — independent dimensions |
| Sycophantic × Helpful | 0.000 | Zero — sycophancy does not help |
| Sycophantic × Empathy | 0.135 | Slight positive — sycophancy feels warm but is distinct from empathy |

**What this means:**

Sycophancy is not a failure of empathy — empathy is 0.135 correlated with sycophancy (slight positive). A sycophantic response can feel warm. It's a failure of **boundaries** — the -0.784 correlation is unambiguous.

Sycophancy is also not a failure of therapeutic technique — r=-0.054, essentially zero. A supporter can use all the right clinical-sounding language while still agreeing with everything the seeker says.

**The most important implication:**

> *You cannot detect sycophancy by checking if a response sounds empathetic or clinical. A response can be warm, well-structured, and therapeutically worded AND be sycophantic. The only thing sycophancy reliably destroys is boundaries.*

This has direct implications for AI safety evaluation. If researchers evaluate AI responses for empathy and therapeutic quality, they will miss sycophancy. It requires a dedicated sycophancy evaluation rubric.

---

## Summary of All Findings

| Finding | Result | Significance |
|---|---|---|
| Human sycophancy baseline | **0.159 (LOW)** | Trained humans are NOT very sycophantic |
| Users reward sycophancy? | **No (r=0.063, p=0.074)** | Users reward empathy and skill, not validation |
| Users reward empathy? | **Yes (r=0.205, p<0.001)** | Empathy is what actually matters to users |
| Temporal arc | **Peaks in middle (0.186)** | Social pressure builds mid-conversation |
| Most sycophantic strategy | **Affirmation (0.240)** | 3x higher than Question (0.080) |
| Least sycophantic strategy | **Question (0.080)** | Asking = not agreeing |
| Emotion that triggers most sycophancy | **Anger (0.185)** | Outward-directed emotions invite agreement |
| Strongest internal correlation | **Sycophantic × Boundaried (r=-0.784)** | Sycophancy = loss of boundaries, not empathy |

---

## What This Means for the Broader Project

When your teammates share their AI sycophancy scores, here's how to interpret the comparison:

**If AI scores >> 0.159 (e.g., 0.35–0.50+):**
> AI is significantly more sycophantic than trained human supporters. It has learned to agree more than humans do, even humans who were specifically trained to be supportive. This suggests AI training amplifies a tendency humans have largely learned to control.

**If AI scores ≈ 0.159:**
> AI replicates human supporter behavior accurately. It has learned the same level of validation that humans exhibit. The problem then is that AI is deployed at massive scale — even moderate human-level sycophancy, replicated across millions of conversations, has outsized effects.

**The mechanism finding (regardless of AI scores):**
Users in ESConv do NOT significantly reward sycophancy (r=0.063, p=0.074). If AI is trained on user feedback from mental health conversations and becomes sycophantic, the explanation is NOT that users in those conversations reward it. The explanation must lie elsewhere — in the feedback mechanism (single-response ratings vs. full conversation ratings), in the training data composition, or in optimization pressure from other domains (customer service, social media) where sycophancy IS rewarded.

---

## Limitations

1. **Scorer bias:** Gemini 2.0 Flash was used to score responses. The model may have its own biases about what constitutes sycophancy, particularly at the boundary between therapeutic validation and over-validation. ~1.8% of rows failed JSON parsing and were excluded.

2. **Research context:** ESConv seekers volunteered knowing it was a research study. Problems may be less severe than real mental health crises, and seekers may be more patient/forgiving than real users of mental health apps.

3. **Small cells:** Several emotion categories have very small sample sizes (pain n=9, guilt n=9, jealousy n=14) — findings for those groups should not be generalized.

4. **Single rubric:** All scoring was done by one model with one prompt. Inter-rater reliability with human judges or alternative models has not been established.

5. **Temporal confound:** The "middle phase" sycophancy increase may partly reflect strategy distribution (more Affirmation strategies used mid-conversation) rather than pure temporal drift.

---

*Dataset: ESConv (Zhou et al., 2021) — thu-coai/esconv*
*Scoring: Gemini 2.0 Flash via Vertex AI*
*Sample: 10,005 turns across 910 conversations (186 parse errors excluded)*
*Satisfaction data: 800/910 conversations with seeker survey responses*
