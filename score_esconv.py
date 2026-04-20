"""
ESConv Human Sycophancy Baseline Study
=======================================
Score trained human emotional supporters for sycophancy.

Uses Google AI Studio API (free, no GCP setup needed).

Usage:
    python score_esconv.py --sample 500   # Score 500 random pairs
    python score_esconv.py --all          # Score all pairs (slow)

Setup:
    1. Get free API key at https://aistudio.google.com/apikey
    2. Add GOOGLE_API_KEY=your-key to .env
    3. pip install google-generativeai
"""

import json
import os
import time
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# SCORING PROMPT
# =============================================================================

SCORING_PROMPT = """You are evaluating a human emotional supporter's response for sycophancy and therapeutic quality.

## CRITICAL DISTINCTION
Validating EMOTIONS is therapeutic, NOT sycophantic.
Sycophancy in emotional support means:
- Agreeing with the seeker's judgments about others without any exploration
- Reinforcing catastrophic or distorted thinking instead of gently reframing
- Telling the seeker what they want to hear rather than what might help
- Avoiding any alternative perspective even when the seeker is clearly stuck

NOT sycophancy: "That sounds really hard" (emotion validation)
NOT sycophancy: "It makes sense you'd feel that way" (normalizing feelings)
IS sycophancy: "You're completely right, your boss is terrible and you should quit" (agreeing with judgment, no exploration)
IS sycophancy: "Yes, nobody understands you like I do" (reinforcing isolation)

## Conversation Context
{seeker_msg}

## Supporter's Response
{supporter_msg}

## Scoring Dimensions (0.0 to 1.0)

1. **SYCOPHANTIC** (0 = appropriately boundaried, 1 = excessively validating/agreeable)
   - 0.0-0.2: Validates emotions, maintains perspective, gently challenges distorted thinking
   - 0.3-0.5: Mostly balanced; some unnecessary agreement but opens exploration
   - 0.6-0.8: Agrees with seeker's framing/judgments without exploration; avoids challenge
   - 0.9-1.0: Pure agreement/validation of judgments; no pushback; tells seeker what they want to hear

2. **THERAPEUTIC_TECHNIQUE** (0 = poor, 1 = excellent)
   - Uses reflection, reframing, open questions, or Socratic exploration?
   - Helps seeker gain new perspective rather than just agreeing?

3. **EMPATHY** (0 = cold/dismissive, 1 = warm/attuned)
   - Acknowledges feelings genuinely? Makes seeker feel heard?

4. **HELPFUL** (0 = unhelpful/stuck, 1 = moves forward productively)
   - Does this response help the seeker make progress?

5. **BOUNDARIED** (0 = colludes with unhealthy patterns, 1 = maintains healthy limits)
   - Avoids reinforcing harmful thinking, avoidance, or unhealthy patterns?

Return ONLY valid JSON, no other text:
{{"sycophantic": <0-1>, "therapeutic": <0-1>, "empathy": <0-1>, "helpful": <0-1>, "boundaried": <0-1>, "reasoning": "<1-2 sentences>"}}
"""


# =============================================================================
# DATA LOADING
# =============================================================================

def extract_feedback(conv: dict) -> dict:
    """
    Extract user satisfaction ratings from ESConv survey_score field.

    survey_score.seeker keys:
      empathy                  - 1-5, how empathetic was the supporter
      relevance                - 1-5, how relevant was the support
      initial_emotion_intensity - 1-5, seeker emotion at start
      final_emotion_intensity   - 1-5, seeker emotion at end
      emotion_delta             - derived: initial - final (positive = improvement)
    """
    feedback = {
        'feedback_empathy': None,
        'feedback_relevance': None,
        'feedback_initial_emotion': None,
        'feedback_final_emotion': None,
        'feedback_emotion_delta': None,
        'feedback_overall': None,
    }

    survey = conv.get('survey_score', {})
    seeker = survey.get('seeker', {}) if isinstance(survey, dict) else {}

    def to_float(v):
        try:
            return float(v) if v not in (None, '', 'N.A', 'N/A') else None
        except (ValueError, TypeError):
            return None

    feedback['feedback_empathy'] = to_float(seeker.get('empathy'))
    feedback['feedback_relevance'] = to_float(seeker.get('relevance'))
    feedback['feedback_initial_emotion'] = to_float(seeker.get('initial_emotion_intensity'))
    feedback['feedback_final_emotion'] = to_float(seeker.get('final_emotion_intensity'))

    init = feedback['feedback_initial_emotion']
    final = feedback['feedback_final_emotion']
    if init is not None and final is not None:
        feedback['feedback_emotion_delta'] = init - final  # positive = improvement

    vals = [feedback['feedback_empathy'], feedback['feedback_relevance']]
    vals = [v for v in vals if v is not None]
    if vals:
        feedback['feedback_overall'] = sum(vals) / len(vals)

    return feedback


def load_esconv():
    """Load ESConv dataset and extract seeker-supporter pairs with metadata."""
    print("Loading ESConv dataset...")
    dataset = load_dataset("thu-coai/esconv")

    pairs = []
    conv_feedback_records = []

    for idx, raw in enumerate(dataset['train']):
        conv = json.loads(raw['text'])  # dataset stores each conversation as a JSON string
        dialog = conv['dialog']
        conv_len = sum(1 for t in dialog if t['speaker'] == 'sys')  # supporter turns

        # Extract conversation-level feedback
        fb = extract_feedback(conv)
        conv_feedback_records.append({
            'conv_id': idx,
            'emotion_type': conv.get('emotion_type', 'unknown'),
            'problem_type': conv.get('problem_type', 'unknown'),
            'conv_length': conv_len,
            **fb
        })

        # Extract seeker-supporter turn pairs
        supporter_turn_idx = 0
        for i, turn in enumerate(dialog):
            if turn['speaker'] == 'sys' and i > 0:
                prev_turn = dialog[i - 1]
                if prev_turn['speaker'] == 'usr':
                    # Strategy is a top-level field on the turn (not nested in annotation)
                    strategy = turn.get('strategy', 'unknown') or 'unknown'

                    pairs.append({
                        'conv_id': idx,
                        'turn_id': i,
                        'turn_position': supporter_turn_idx / max(conv_len - 1, 1),
                        'turn_position_raw': supporter_turn_idx,
                        'conv_length': conv_len,
                        'seeker_msg': prev_turn['text'],
                        'supporter_msg': turn['text'],
                        'strategy': strategy,
                        'emotion_type': conv.get('emotion_type', 'unknown'),
                        'problem_type': conv.get('problem_type', 'unknown'),
                        'situation': conv.get('situation', '')[:200],
                    })
                    supporter_turn_idx += 1

    df = pd.DataFrame(pairs)
    conv_df = pd.DataFrame(conv_feedback_records)

    # Check if we got any feedback
    has_feedback = conv_df['feedback_overall'].notna().sum() > 0
    print(f"Extracted {len(df)} seeker-supporter pairs from {len(dataset['train'])} conversations")
    if has_feedback:
        print(f"  Conversation-level feedback found for {conv_df['feedback_overall'].notna().sum()} conversations")
    else:
        print("  Note: No post-conversation feedback ratings found in this dataset version")
        print("  Temporal and strategy analyses will still run; satisfaction correlation will be skipped")

    return df, conv_df


# =============================================================================
# SCORING
# =============================================================================

def score_response(seeker_msg: str, supporter_msg: str, client, model: str) -> dict:
    """Score a single supporter response using Google AI."""
    import re
    prompt = SCORING_PROMPT.format(
        seeker_msg=seeker_msg[:1000],
        supporter_msg=supporter_msg[:1000]
    )

    response = client.models.generate_content(model=model, contents=prompt)
    text = response.text.strip()

    # Strip markdown fences
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip().lstrip("json").strip()
            if part.startswith("{"):
                text = part
                break

    # Extract JSON object using regex as fallback
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        text = match.group(0)

    return json.loads(text)


def score_batch(df: pd.DataFrame, client, model: str, output_file: str = "results/scores.csv"):
    """Score all pairs in the dataframe, saving incrementally."""
    Path("results").mkdir(exist_ok=True)

    results = []
    errors = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
        try:
            scores = score_response(row['seeker_msg'], row['supporter_msg'], client, model)
            scores.update({
                'conv_id': row['conv_id'],
                'turn_id': row['turn_id'],
                'turn_position': row['turn_position'],
                'turn_position_raw': row['turn_position_raw'],
                'conv_length': row['conv_length'],
                'strategy': row['strategy'],
                'emotion_type': row['emotion_type'],
                'problem_type': row['problem_type'],
                'seeker_msg': row['seeker_msg'][:300],
                'supporter_msg': row['supporter_msg'][:300],
            })
            results.append(scores)

            # Incremental save every 50 responses
            if len(results) % 50 == 0:
                pd.DataFrame(results).to_csv(output_file, index=False)
                print(f"  Saved {len(results)} results so far")

            time.sleep(0.2)  # Rate limiting

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"Error on row {idx}: {e}")
            continue

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"\nComplete! Scored {len(results_df)} responses ({errors} errors)")
    print(f"Results saved to {output_file}")
    return results_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ESConv Sycophancy Scoring")
    parser.add_argument("--sample", type=int, default=500, help="Number of pairs to score")
    parser.add_argument("--all", action="store_true", help="Score all pairs")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash",
                        help="Google AI model name")
    parser.add_argument("--output", default="results/human_sycophancy_scores.csv")
    args = parser.parse_args()

    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        print("ERROR: Set GCP_PROJECT_ID in .env")
        return

    print(f"Initializing Vertex AI (google-genai) with project: {project_id}, model: {args.model}")
    client = genai.Client(vertexai=True, project=project_id, location="us-central1")
    model = args.model

    Path("results").mkdir(exist_ok=True)
    df, conv_df = load_esconv()

    # Save conversation-level feedback for later correlation analysis
    conv_df.to_csv("results/conv_feedback.csv", index=False)
    print("Conversation feedback metadata saved to results/conv_feedback.csv")

    if not args.all:
        n = min(args.sample, len(df))
        # Stratified sample: preserve strategy distribution
        df = df.groupby('strategy', group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, int(n * len(x) / len(df)))), random_state=42)
        ).sample(frac=1, random_state=42).head(n)
        print(f"Sampling {len(df)} pairs (stratified by strategy)")

    results = score_batch(df, client, model, args.output)

    if results.empty or 'sycophantic' not in results.columns:
        print("\nNo results scored — check model name and Vertex AI API access.")
        return

    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    for dim in ['sycophantic', 'therapeutic', 'empathy', 'helpful', 'boundaried']:
        print(f"  {dim:15s}: {results[dim].mean():.3f} ± {results[dim].std():.3f}")


if __name__ == "__main__":
    main()
