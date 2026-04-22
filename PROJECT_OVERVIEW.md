# Project Overview: Human Sycophancy Baseline in Mental Health Support

---

## What We Did (Completed)

We went through **910 real conversations** (10,005 scored turns) where a trained human volunteer helped someone in emotional distress (stressed about job, relationship, anxiety etc.).

For each conversation turn, we asked Gemini: *"Did the helper just agree with everything the person said, or did they actually help?"*

Gemini scored each response 0–1 on 5 dimensions. The key one is **sycophancy** — did the supporter just tell the person what they wanted to hear?

### What We Found

1. **Human supporters are NOT very sycophantic** — average score = 0.159 (LOW). Most responses cluster near 0.
2. **Users do NOT reward sycophancy** — r = 0.063, p = 0.074 (not significant). Users reward empathy and helpfulness instead.
3. **Sycophancy peaks mid-conversation** — it rises in the middle (0.186) when rapport is established, then drops again.
4. **Affirmation strategy = most sycophantic (0.240), Question = least (0.080)** — threefold difference, but equal empathy.

### Why It Matters

Our teammates are measuring how sycophantic **AI chatbots** are. We measured how sycophantic **trained humans** are.

The refined punchline based on actual findings:

> *"Trained human supporters maintain low sycophancy (0.159) and users don't reward it. If AI systems trained on human preferences are MORE sycophantic than this, the problem isn't just learned human behavior — AI training is amplifying a tendency that humans have largely learned to suppress."*

That's a publishable finding.

---

## The Dataset: ESConv

### What It Actually Is

**Real conversations, not synthetic.**

### How It Was Collected

Researchers at Tsinghua University recruited real people on a crowdsourcing platform (like MTurk). They paired them up:

- **Seeker** — a real person with a real emotional problem (job stress, relationship issues, anxiety, grief, etc.). They were asked to talk about something genuinely bothering them.
- **Supporter** — a different real person who was given training materials on emotional support techniques before the conversation.

Both people were anonymous. The conversations happened live, in real time, via text chat.

### What's In It

- **910 conversations**, 10,005 scored supporter turns
- Each supporter turn is labeled with which strategy they used (see strategy list below)
- After the conversation, the **seeker filled out a survey** rating how empathetic, relevant, and helpful the support was, and how their emotional state changed
- Problem types include: job crisis, academic pressure, breakup with partner, problems with friends, conflict with parents, Issues with Parents, Issues with Children, ongoing depression, Sleep Problems, Procrastination, School Bullying, Appearance Anxiety, Alcohol Abuse

### Every Field in the Dataset Explained

| Field | What It Means |
|---|---|
| `emotion_type` | What emotion the seeker was feeling — anxiety, sadness, anger, depression, etc. |
| `problem_type` | Category of their problem — job crisis, relationship, academic pressure, etc. |
| `experience_type` | Whether the problem is happening **right now** (Current Experience) or happened **in the past** (Previous Experience) |
| `situation` | The seeker's own description of their problem in a few sentences |
| `speaker` | Who sent the message — SEEKER (person with the problem) or SUPPORTER (trained volunteer helping) |
| `strategy` | Which support technique the supporter used on that turn (only applies to supporter turns) |
| `message` | The actual text of what was said |
| `seeker_survey_empathy` | Seeker rates: did the supporter understand how I felt? (1-5, higher = better) |
| `seeker_survey_relevance` | Seeker rates: was the support relevant to my problem? (1-5, higher = better) |
| `seeker_survey_initial_emotion` | How distressed the seeker felt BEFORE the conversation (1-5, higher = more distressed) |
| `seeker_survey_final_emotion` | How distressed the seeker felt AFTER the conversation (1-5, higher = more distressed) |
| `supporter_survey_relevance` | Supporter's own rating of how relevant they felt their support was |
| `seeker_question1/2` | Open-ended feedback from the seeker about the crowdsourcing platform itself — NOT about the conversation. Things like "did you enjoy the task?" We do not use these in our analysis. |
| `supporter_question1/2` | Same but from the supporter — platform feedback, not used in our analysis |

**Note on emotion scores:** Lower final emotion = GOOD. It means the seeker felt less distressed after the conversation. We calculate `emotion_delta = initial - final`. A positive delta means the conversation helped.

### The 8 Support Strategies Explained

These are the techniques the supporter could use on each turn:

| Strategy | What It Means in Plain English |
|---|---|
| **Question** | Asking the seeker a question to help them think through their situation |
| **Restatement or Paraphrasing** | Repeating back what the seeker said in different words to show understanding |
| **Reflection of Feelings** | Naming and acknowledging the seeker's emotions ("it sounds like you're feeling overwhelmed") |
| **Self-disclosure** | The supporter sharing their own similar experience ("I've felt that way too") |
| **Affirmation and Reassurance** | Validating the seeker's strengths or feelings ("you're doing the right thing", "your feelings are valid") |
| **Providing Suggestions** | Offering practical advice or next steps |
| **Information** | Sharing factual information relevant to the seeker's problem |
| **Others** | Anything that doesn't fit the above categories |

### Why It's Valuable for Our Study

- The supporters were **trained volunteers, not professionals** — similar to how AI is trained on guidelines
- The seeker ratings give us an **outcome measure** (did the support actually help emotionally?)
- It's the closest real-world analog to how AI chatbots are deployed for mental health support

### One Caveat

The seekers volunteered to discuss their problems knowing it was a research study, so the problems may be slightly less severe than real crisis situations. But the emotions and conversations are genuine — not scripted.

---

## How This Fits the Broader Project

| Component | Dataset | Purpose |
|---|---|---|
| AI sycophancy scoring | Kaggle: Human & LLM Mental Health Conversations | Teammates |
| Sycophancy training data analysis | HuggingFace: sycofact-training-data (42K examples) | Teammates |
| **Human baseline (this study)** | **ESConv: 910 real human support conversations** | **This repo** |

The joint finding: *"Trained human supporters show low sycophancy (0.159) and users don't reward it. If AI systems trained via RLHF are more sycophantic, it is an amplification beyond the human baseline, not simply a learned replication of it."*
