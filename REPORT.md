# Human Sycophancy in Emotional Support Conversations: A Baseline Study
### ESConv Dataset, 10,005 Scored Turns, 910 Conversations

---

## Abstract

This study establishes a human sycophancy baseline in trained emotional support conversations using the ESConv dataset. We scored 10,005 supporter turns across 910 real human-human conversations on five dimensions (sycophancy, therapeutic technique, empathy, helpfulness, and boundary maintenance), using Gemini 2.0 Flash via Vertex AI. We find that trained human supporters exhibit low sycophancy (mean = 0.159), that users do not significantly reward sycophantic responses (r = 0.063, p = 0.074), and that sycophancy is most strongly associated with boundary erosion (r = -0.784) rather than with empathy or therapeutic quality. These findings establish a human reference point for evaluating AI chatbot behavior in mental health contexts and suggest that AI sycophancy, if observed to exceed this baseline, represents an amplification of human tendencies rather than just a straightforward replication of them.

---

## 1. Introduction

Sycophancy in AI systems refers to the tendency to tell people what they want to hear rather than what might actually help them. In mental health support contexts, this is especially concerning because AI chatbots that over-validate users, agree with distorted thinking, and avoid challenge may provide momentarily satisfying interactions while failing to support genuine wellbeing.

A gap in the existing literature is that there is no clear reference point for how sycophantic trained human supporters actually are. If humans who have been specifically trained in emotional support techniques also exhibit sycophantic behavior, then AI systems that learn from human data may simply be replicating, or even amplifying, a pattern that already exists in the training source. Understanding the human baseline is therefore important for interpreting AI behavior and for designing better training objectives.

This study addresses that gap by systematically scoring trained human emotional supporters in the ESConv dataset and correlating those scores with seeker-reported satisfaction outcomes. The central questions are: (1) How sycophantic are trained human supporters? (2) Do users reward sycophancy? (3) Which conversation features, including strategies, emotions, and temporal position, are most associated with sycophantic behavior?

---

## 2. Dataset and Methods

### 2.1 The ESConv Dataset

The Emotional Support Conversation (ESConv) dataset (Zhou et al., 2021) contains 910 real conversations between trained volunteer supporters and individuals seeking emotional help, collected via a crowdsourcing platform. Supporters were given training materials on emotional support strategies before conversations began. This makes ESConv particularly well-suited for a human baseline study because the supporters are trained but not professional therapists, which is roughly analogous to AI systems that are trained on guidelines but not clinically licensed.

Each conversation includes turn-by-turn strategy annotations across eight categories (Question, Restatement or Paraphrasing, Reflection of Feelings, Self-disclosure, Affirmation and Reassurance, Providing Suggestions, Information, and Others), as well as conversation-level metadata including the seeker's emotion type and problem type. Importantly, 800 of the 910 conversations include a post-conversation survey from the seeker rating the quality of support on dimensions including empathy, relevance, and emotional change (initial vs. final emotional intensity). We extracted 10,191 seeker-supporter turn pairs from the training split; 10,005 were successfully scored (186 parse errors excluded, 1.8%).

### 2.2 Scoring Rubric

A critical methodological challenge in scoring sycophancy in mental health contexts is distinguishing it from therapeutic validation. Validating a seeker's emotions ("that sounds really painful") is good practice and not sycophantic. Agreeing with a seeker's judgments without any exploration ("you're right, your boss is terrible") is sycophantic. Our scoring prompt explicitly operationalized this distinction.

Each supporter turn was scored 0.0 to 1.0 on five dimensions using Gemini 2.0 Flash via Vertex AI:

- **Sycophantic**: 0 = appropriately challenges distorted thinking, 1 = pure agreement/validation of judgments
- **Therapeutic**: 0 = no technique, 1 = uses reflection, reframing, Socratic questioning
- **Empathy**: 0 = cold or dismissive, 1 = warm and attuned to feelings
- **Helpful**: 0 = unproductive, 1 = moves conversation forward
- **Boundaried**: 0 = colludes with unhealthy patterns, 1 = maintains healthy limits

### 2.3 Satisfaction Correlation

Turn-level scores were aggregated to conversation level (mean per conversation) and merged with the seeker's post-conversation survey. The seeker's overall satisfaction rating (mean of empathy and relevance ratings, on a 1 to 5 scale) was used as the primary outcome variable. Emotion delta (initial minus final emotional intensity) was used as a secondary outcome capturing actual therapeutic effectiveness. Pearson correlations were computed between each score dimension and each satisfaction metric across the 800 conversations with available feedback.

---

## 3. Results

### 3.1 Overall Human Sycophancy Baseline

The mean sycophancy score across all 10,005 scored turns was **0.159 (SD = 0.166, median = 0.100)**. This is classified as LOW on our rubric. The distribution is heavily right-skewed: 71.5% of responses scored below 0.2, and only 6.0% exceeded 0.5. Fewer than 1% of responses (0.7%) exceeded 0.7, which is the threshold for clearly problematic over-validation. This indicates that trained human supporters, most of the time, are not simply agreeing with everything the seeker says. The full score profile was: therapeutic = 0.311, empathy = 0.602, helpful = 0.407, boundaried = 0.782. The pattern of high empathy and strong boundaries alongside low sycophancy is roughly what well-trained emotional support should theoretically look like.

### 3.2 Satisfaction Correlation: Users Do Not Reward Sycophancy

The correlation between conversation-level sycophancy and overall user satisfaction was r = 0.063 (p = 0.074, not significant). This is the study's central finding. Users do not significantly rate sycophantic conversations as more satisfying. The predictors that do significantly predict satisfaction are empathy (r = 0.205, p < 0.001), helpfulness (r = 0.182, p < 0.001), and therapeutic technique (r = 0.167, p < 0.001). Maintaining good boundaries showed no significant relationship with satisfaction (r = 0.017, p = 0.627), which suggests users are largely unaware of whether the supporter challenged their thinking appropriately.

To assess the combined predictive power of all five dimensions simultaneously, we ran a multivariate linear regression predicting user satisfaction from all five score dimensions (standardized). The model achieved R² = 0.048 (cross-validated R² = 0.030), meaning that the five dimensions together explain less than 5% of variance in user satisfaction. This is itself a meaningful finding: user satisfaction in emotional support conversations is dominated by subjective, idiosyncratic factors rather than objectively measurable response quality. Notably, the sycophantic coefficient was positive (beta = 0.083) in the multivariate model. When controlling for empathy and helpfulness, sycophancy shows a slight positive association with satisfaction, which is consistent with users partially conflating validation with warmth.

This is not a null result. It is a positive finding with important implications: trained human supporters have learned that sycophancy does not earn better ratings. They have learned to distinguish genuine empathy, which users do reward, from empty validation, which users neither particularly reward nor punish. The question this raises for AI is why systems trained on human feedback still become sycophantic if the human feedback signal does not reward it.

Notably, no dimension significantly predicted actual emotional improvement (emotion delta). Empathy came closest (r = 0.027, p = 0.449) but remained non-significant. This suggests that user satisfaction ratings and actual therapeutic effectiveness may be partially decoupled, with users rating supportive-feeling conversations highly regardless of whether their emotional state meaningfully improved.

### 3.3 Temporal Arc: Sycophancy Peaks Mid-Conversation

Binning turns into early (turns 0 to 33%), middle (turns 33 to 67%), and late (turns 67 to 100%) phases revealed a significant temporal arc (ANOVA: F = 92.18, p < 0.001). Sycophancy was lowest in the early phase (mean = 0.137), peaked in the middle phase (mean = 0.186), and partially recovered in the late phase (mean = 0.150). Correspondingly, the boundaried score was highest in the early phase (0.809), dropped to its lowest in the middle phase (0.747), and recovered late (0.802).

The most plausible interpretation is that the early phase is dominated by information-gathering behaviors such as clarifying questions and restatements, which are structurally incompatible with sycophancy. The middle phase, once rapport is established, creates social pressure to validate the seeker's perspective. The late phase shifts toward practical support including suggestions, information, and next steps, which is again less validation-oriented. The therapeutic score also decreases substantially in the late phase (from 0.336 to 0.217), suggesting that late-conversation responses tend to be simpler and more direct.

### 3.4 Strategy-Level Differences: Questioning Is the Antidote to Sycophancy

The largest effect in the study was found at the level of support strategy (ANOVA: F = 161.01, p < 0.001). Sycophancy varied from 0.080 for Question-based responses to 0.240 for Affirmation and Reassurance, which is a threefold difference (t = 31.19, p < 0.001, Cohen's d = 1.053, a large effect by conventional standards). The full ranking from highest to lowest sycophancy was: Affirmation and Reassurance (0.240), Self-disclosure (0.210), Reflection of Feelings (0.204), Providing Suggestions (0.151), Restatement or Paraphrasing (0.148), Information (0.146), Others (0.130), Question (0.080).

The reason for this pattern is largely structural. Affirmation and Reassurance is designed to validate the seeker's feelings and strengths, making it agreement-oriented by definition. Self-disclosure ("I've felt that way too") creates social pressure to align with the seeker's framing. Questioning, by contrast, invites the seeker to examine their own thinking, and it is essentially impossible to simultaneously ask "why do you think that?" and agree with everything the person says.

A particularly important finding emerges from comparing Affirmation and Question directly. Their empathy scores are nearly identical (0.628 vs. 0.625), but their sycophancy scores differ threefold (0.240 vs. 0.080) and their boundaried scores differ substantially (0.695 vs. 0.874). This demonstrates that empathy and sycophancy are separable dimensions. A supporter can be equally warm and attuned using either strategy, but one maintains healthy limits and the other does not. The choice of strategy, not the level of warmth, is what drives sycophancy.

### 3.5 Emotion-Driven Differences: Outward Emotions Trigger More Sycophancy

Supporters were more sycophantic when responding to outward-directed emotions (anger: 0.185, disgust: 0.185) than to inward-directed emotions (depression: 0.151, anxiety: 0.155, shame: 0.154). The three largest emotion categories by count, which were anxiety (n = 2,680), depression (n = 2,644), and sadness (n = 2,413), all showed below-average sycophancy (0.155, 0.151, 0.162 respectively), while anger (n = 827) showed the highest sycophancy among adequately sampled groups (0.185).

The explanation is largely social-cognitive. When a seeker is angry at a boss, partner, or situation, it is socially natural to agree with their narrative and to validate both their emotion and their judgment of the external target. When a seeker is depressed or anxious, there is no external target to agree about and the supporter must engage with the emotion itself, which requires exploration rather than agreement.

### 3.6 Internal Validity: Sycophancy Destroys Boundaries, Not Empathy

The correlation matrix reveals the clearest internal relationship in the study: sycophantic x boundaried = -0.784 (p < 0.001). These two dimensions are nearly mirror opposites. As sycophancy increases, boundary maintenance decreases at a near-linear rate. By contrast, sycophantic x therapeutic = -0.054 (essentially zero) and sycophantic x empathy = 0.135 (slight positive).

This has an important implication: sycophancy is not a failure of empathy, and it does not replace therapeutic quality. It is specifically a failure of boundaries. A sycophantic response can feel warm (empathy correlation = 0.135) and can even sound therapeutically structured (therapeutic correlation = -0.054, near zero). The only thing sycophancy reliably undermines is the supporter's ability to maintain a non-colluding, appropriately challenging stance.

For AI evaluation specifically, this means that reviewing responses for empathetic tone or therapeutic language is not sufficient to detect sycophancy. Dedicated evaluation of boundary maintenance, including whether the response challenges distorted thinking, avoids colluding with harmful narratives, and offers alternative perspectives, is necessary.

---

## 4. Discussion

### 4.1 The Human Baseline in Context

The human sycophancy baseline of 0.159 provides a concrete reference point for AI comparison. If AI chatbots deployed in mental health contexts score significantly above this threshold, it suggests that AI training has amplified a tendency that human training has largely suppressed. If AI scores are close to the human baseline, it suggests that AI has successfully replicated human supporter behavior, for better and for worse.

The finding that users do not significantly reward sycophancy (r = 0.063, p = 0.074) adds nuance to the standard narrative about RLHF and sycophancy. The standard narrative holds that AI becomes sycophantic because human feedback rewards it. Our data suggests this may be incomplete. In the context of extended support conversations, users do not reward sycophancy. If AI systems trained on human preferences in similar contexts still become sycophantic, the cause must be sought elsewhere, perhaps in the type of feedback collected (single-response preference ratings vs. full-conversation satisfaction surveys), in the composition of training data (domains where sycophancy is rewarded, such as customer service), or in the mechanics of preference optimization itself.

### 4.2 Strategy as a Design Variable

The strategy-level findings suggest a concrete intervention: if AI systems default to Question-based responses rather than Affirmation-heavy responses, sycophancy would structurally decrease without any loss of empathy. This is not a theoretical claim. The data shows directly that the two strategies produce equivalent empathy scores (0.625 vs. 0.628) with dramatically different sycophancy scores (0.080 vs. 0.240). Training AI to ask more and affirm less may be one of the more tractable paths to reducing sycophancy in mental health applications.

### 4.3 Limitations

Several limitations should be noted. First, all scoring was performed by a single model (Gemini 2.0 Flash) with a single prompt. Inter-rater reliability with human judges or alternative scoring models has not been established, and the model may have systematic biases in how it interprets the sycophancy rubric. Second, ESConv was collected in a research context, and seekers may behave differently from users of deployed mental health apps since the stakes and emotional intensity may differ. Third, several emotion categories had very small sample sizes (pain n = 9, guilt n = 9, jealousy n = 14), and findings for those groups should not be generalized. Fourth, the temporal findings may be partially confounded by strategy distribution. If Affirmation strategies are used more frequently in the middle phase, the temporal arc may partly reflect strategy choice rather than pure temporal drift.

---

## 5. Conclusion

Trained human emotional supporters exhibit low sycophancy (mean = 0.159) and are not significantly rewarded for it by users. Sycophancy in this population is most strongly associated with boundary erosion (r = -0.784), is three times higher in affirmation-based strategies than questioning-based ones, peaks in the middle phase of conversations, and is most pronounced when seekers express outward-directed emotions like anger and disgust. These findings establish a concrete human reference point for evaluating AI chatbot sycophancy, suggest that AI systems trained on similar data should not necessarily become sycophantic, and identify strategy choice as a tractable lever for sycophancy reduction.

---

## 6. Policy Implications

### 6.1 Reframe the Problem: Sycophancy Is a Design Choice, Not an Inevitable Outcome

The dominant narrative in AI policy treats sycophancy as a byproduct of training on human feedback, an unavoidable consequence of optimizing for user approval. Our findings challenge this framing. Trained human supporters, who also operate under social pressure to validate the people they help, maintain a sycophancy level of only 0.159. Users do not significantly reward sycophancy in this context (r = 0.063, p = 0.074). If human supporters can learn to be warm without being sycophantic, and users do not particularly reward sycophancy when they receive it, then AI sycophancy is not inevitable. It is a failure of training design. Policymakers and regulators evaluating AI mental health tools should hold developers accountable to this standard: trained humans achieve low sycophancy at scale, and AI systems should be expected to do the same.

### 6.2 Mandatory Sycophancy Evaluation for AI Mental Health Tools

Currently, AI mental health applications are evaluated primarily for safety (does it say harmful things?), empathy (does it sound warm?), and clinical accuracy (does it give correct information?). Our findings show that sycophancy is distinct from all three of these dimensions. A response can be safe, warm, and accurate while still being sycophantic, in the sense of agreeing with distorted thinking, reinforcing avoidance, and failing to gently challenge harmful narratives. Regulatory frameworks governing AI in mental health, including any future FDA guidance on AI-based wellness applications, FTC rules on deceptive AI practices, or EU AI Act provisions on high-risk AI in health contexts, should require explicit sycophancy evaluation as a distinct safety dimension. Evaluating empathy alone is not sufficient.

### 6.3 Strategy-Based Design Standards

Our finding that Question-based responses show threefold lower sycophancy than Affirmation-based responses (0.080 vs. 0.240) with equivalent empathy (0.625 vs. 0.628) has direct implications for how AI mental health tools should be designed and audited. Platform developers should be required to audit their systems for strategy distribution. If an AI chatbot is overwhelmingly affirmation-heavy and rarely poses exploratory questions, that is a measurable proxy for sycophancy risk. Regulators could mandate that developers disclose the distribution of response strategies used by their systems, similar to how algorithmic transparency is required in other high-risk domains. This is a concrete, auditable standard that does not require access to model weights or training data.

### 6.4 Protect Vulnerable Populations from Sycophancy Drift

Our temporal analysis shows that sycophancy peaks in the middle phase of conversations (0.186), when rapport has been established. For human supporters, this is a manageable and recoverable pattern since they pull back in the late phase. For AI systems deployed for chronic mental health support, where users interact repeatedly over weeks or months, this drift may accumulate rather than recover. A user with depression who interacts with an AI daily could receive increasingly validating responses over time, reinforcing rather than challenging their negative self-narrative. Policies governing AI in mental health should require longitudinal evaluation of sycophancy drift and not just snapshot audits of individual responses. They should also mandate mechanisms for recalibration when sustained over-validation is detected.

### 6.5 Anger-Triggered Sycophancy and the Risk of Radicalization

Our emotion analysis shows that anger triggers the highest sycophancy among adequately-sampled emotions (0.185, n = 827). When someone expresses anger at a person, institution, or situation, supporters, both human and likely AI, are most prone to agreeing with their framing without exploration. At the scale of millions of AI interactions, this has implications beyond mental health. An AI that consistently validates anger without questioning it could contribute to the entrenchment of grievances, misattribution of blame, and in extreme cases, radicalization. Policymakers working on both mental health AI and broader AI safety should recognize that sycophancy in anger contexts is not just a therapeutic failure. It is a potential vector for harm at population scale. Specific provisions requiring AI systems to probe and explore anger-related narratives rather than simply validate them should be considered in any high-risk AI regulation.

### 6.6 The Feedback Mechanism Problem

Our finding that users do not significantly reward sycophancy in extended support conversations (r = 0.063, p = 0.074) raises a critical policy question: if the human feedback signal does not reward sycophancy in therapeutic contexts, why do AI systems trained on human feedback become sycophantic? The likely answer is that AI systems are not trained solely on mental health conversation feedback. They are trained on preference data from customer service, social media, and general conversation, where sycophancy likely is rewarded. This cross-domain contamination of training signals is a governance problem. Policymakers should require that AI systems deployed in mental health contexts are fine-tuned or evaluated specifically on mental health conversation data, and that developers disclose what feedback domains were used in training. A chatbot trained to please customers may be fundamentally unsuited for therapeutic support, regardless of its general capabilities.

### 6.7 Summary of Policy Recommendations

| Recommendation | Target | Basis in Findings |
|---|---|---|
| Require explicit sycophancy evaluation for AI mental health tools | Regulators, FDA, EU AI Act | Sycophancy is distinct from empathy and safety; cannot be detected by existing metrics |
| Mandate strategy distribution disclosure | Platform developers | Affirmation-heavy systems are structurally more sycophantic (0.240 vs. 0.080) |
| Require longitudinal sycophancy audits, not just snapshot evaluations | AI auditors, platform developers | Sycophancy peaks mid-conversation; drift over repeated interactions is untested |
| Develop specific standards for anger-triggered sycophancy | Regulators, safety researchers | Anger triggers highest sycophancy (0.185); population-scale risk of grievance reinforcement |
| Require disclosure of training feedback domains for health AI | Regulators, FTC | Cross-domain training signal contamination may drive sycophancy in health contexts |
| Hold AI to the human baseline (0.159) as a minimum standard | Developers, procurement bodies | Trained humans achieve this; AI should not fall below it |

---

## References

Zhou, K., Peng, M., Zheng, Y., & Huang, M. (2021). Towards emotional support dialog systems. *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics*, 3469-3483.

---

*Data: ESConv (thu-coai/esconv) | Scoring: Gemini 2.0 Flash via Vertex AI*
*N = 10,005 turns, 910 conversations, 800 with satisfaction ratings*
*All statistics verified against raw data files in results/*
