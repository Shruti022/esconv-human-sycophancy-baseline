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

- **910 conversations**, ~10,000 turns total
- Each supporter turn is labeled with which strategy they used (Question, Affirmation, Restatement, etc.)
- After the conversation, the **seeker filled out a survey** rating how empathetic, relevant, and helpful the support was — and how their emotional state changed
- Problem types include: job crisis, relationship issues, academic pressure, health issues, family conflict, and more

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
