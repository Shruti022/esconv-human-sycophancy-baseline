"""
ESConv Analysis Script
======================
Analyze scored human supporter responses.

Usage:
    python analyze.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path


def load_results(filepath="results/human_sycophancy_scores.csv"):
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} scored responses from {df['conv_id'].nunique()} conversations")
    return df


def load_conv_feedback(filepath="results/conv_feedback.csv"):
    if not Path(filepath).exists():
        return None
    df = pd.read_csv(filepath)
    has_feedback = df['feedback_overall'].notna().sum() > 0
    if not has_feedback:
        print("Note: No user satisfaction ratings in dataset — skipping satisfaction correlation")
        return None
    print(f"Loaded conversation feedback for {df['feedback_overall'].notna().sum()} conversations")
    return df


def overall_summary(df):
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY: HUMAN SUPPORTER SYCOPHANCY BASELINE")
    print("=" * 70)

    dimensions = ['sycophantic', 'therapeutic', 'empathy', 'helpful', 'boundaried']
    print("\nDimension Statistics:")
    print("-" * 60)
    for dim in dimensions:
        mean = df[dim].mean()
        std = df[dim].std()
        median = df[dim].median()
        print(f"  {dim:15s}: mean={mean:.3f}, std={std:.3f}, median={median:.3f}")

    syc_mean = df['sycophantic'].mean()
    if syc_mean < 0.3:
        interpretation = "LOW — human supporters maintain appropriate limits"
    elif syc_mean < 0.5:
        interpretation = "MODERATE — some validation bias present"
    else:
        interpretation = "HIGH — significant over-validation detected"

    print(f"\nKEY FINDING: Human sycophancy baseline = {syc_mean:.3f}")
    print(f"Interpretation: {interpretation}")


def satisfaction_correlation(scores_df, conv_df):
    """
    THE PUBLISHABLE ANALYSIS:
    Does higher conversation-level sycophancy predict better user satisfaction ratings?
    If yes: users reward sycophancy — this is the reward signal that trains AI to over-validate.
    """
    print("\n" + "=" * 70)
    print("SATISFACTION CORRELATION (Key Finding)")
    print("=" * 70)
    print("Question: Do sycophantic responses get rated as MORE helpful by users?")

    # Aggregate turn scores to conversation level
    conv_scores = scores_df.groupby('conv_id').agg({
        'sycophantic': 'mean',
        'therapeutic': 'mean',
        'empathy': 'mean',
        'helpful': 'mean',
        'boundaried': 'mean',
    }).reset_index()

    # Merge with user feedback
    fb_cols = [c for c in ['feedback_overall', 'feedback_empathy', 'feedback_relevance',
                            'feedback_emotion_delta', 'feedback_initial_emotion',
                            'feedback_final_emotion']
               if c in conv_df.columns]
    merged = conv_scores.merge(conv_df[['conv_id'] + fb_cols], on='conv_id', how='inner')
    merged = merged.dropna(subset=['feedback_overall'])

    if len(merged) < 10:
        print(f"  Too few conversations with feedback ({len(merged)}) — skipping")
        return None

    print(f"\n  Analyzing {len(merged)} conversations with both scores and user ratings\n")

    score_dims = ['sycophantic', 'therapeutic', 'empathy', 'helpful', 'boundaried']
    feedback_dims = [c for c in ['feedback_overall', 'feedback_empathy', 'feedback_relevance',
                                  'feedback_emotion_delta']
                     if c in merged.columns and merged[c].notna().sum() >= 10]

    print("  Correlations with user satisfaction ratings:")
    print(f"  {'Score Dimension':20s}  {'Feedback Metric':25s}  {'r':>7}  {'p':>8}  {'sig':>5}")
    print("  " + "-" * 70)

    results = []
    for sdim in score_dims:
        for fdim in feedback_dims:
            sub = merged[[sdim, fdim]].dropna()
            if len(sub) < 10:
                continue
            r, p = stats.pearsonr(sub[sdim], sub[fdim])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {sdim:20s}  {fdim:25s}  {r:7.3f}  {p:8.4f}  {sig:>5}")
            results.append({'score': sdim, 'feedback': fdim, 'r': r, 'p': p})

    results_df = pd.DataFrame(results)

    # Highlight the headline finding
    syc_overall = results_df[
        (results_df['score'] == 'sycophantic') &
        (results_df['feedback'] == 'feedback_overall')
    ]
    if not syc_overall.empty:
        r = syc_overall.iloc[0]['r']
        p = syc_overall.iloc[0]['p']
        print(f"\n  HEADLINE: sycophancy × user satisfaction: r={r:.3f}, p={p:.4f}")
        if r > 0 and p < 0.05:
            print("  => Users rate sycophantic responses as MORE helpful.")
            print("     This is the reward signal that trains AI to over-validate.")
        elif r < 0 and p < 0.05:
            print("  => Users rate sycophantic responses as LESS helpful.")
            print("     Trained human supporters have learned to resist the sycophancy trap.")
        else:
            print("  => No significant correlation between sycophancy and user satisfaction.")

    merged.to_csv("results/conv_level_analysis.csv", index=False)
    return merged


def temporal_analysis(df):
    """
    Does sycophancy increase as conversations progress?
    Early turns vs. late turns — does rapport-building lead to over-validation?
    """
    print("\n" + "=" * 70)
    print("TEMPORAL ANALYSIS: Sycophancy Over Conversation Arc")
    print("=" * 70)

    # Bin into thirds: early / middle / late
    df = df.copy()
    df['phase'] = pd.cut(df['turn_position'], bins=[0, 0.33, 0.67, 1.0],
                          labels=['early', 'middle', 'late'], include_lowest=True)

    phase_stats = df.groupby('phase', observed=True).agg({
        'sycophantic': ['mean', 'std', 'count'],
        'therapeutic': 'mean',
        'boundaried': 'mean',
    }).round(3)
    phase_stats.columns = ['syc_mean', 'syc_std', 'count', 'therapeutic', 'boundaried']

    print("\nSycophancy by conversation phase:")
    print(phase_stats.to_string())

    # ANOVA across phases
    early = df[df['phase'] == 'early']['sycophantic'].dropna()
    middle = df[df['phase'] == 'middle']['sycophantic'].dropna()
    late = df[df['phase'] == 'late']['sycophantic'].dropna()

    if len(early) >= 5 and len(late) >= 5:
        f_stat, p_value = stats.f_oneway(early, middle, late)
        print(f"\n  ANOVA (sycophancy ~ phase): F={f_stat:.3f}, p={p_value:.4f}")
        if p_value < 0.05:
            direction = "increases" if late.mean() > early.mean() else "decreases"
            print(f"  => Sycophancy {direction} as conversations progress "
                  f"(early={early.mean():.3f} → late={late.mean():.3f})")
        else:
            print("  => No significant change in sycophancy across conversation phases")

    phase_stats.to_csv("results/temporal_analysis.csv")
    return phase_stats


def by_strategy(df):
    print("\n" + "=" * 70)
    print("SYCOPHANCY BY SUPPORT STRATEGY")
    print("=" * 70)

    strategy_stats = df.groupby('strategy').agg({
        'sycophantic': ['mean', 'std', 'count'],
        'therapeutic': 'mean',
        'empathy': 'mean',
    }).round(3)
    strategy_stats.columns = ['syc_mean', 'syc_std', 'count', 'therapeutic', 'empathy']
    strategy_stats = strategy_stats[strategy_stats['count'] >= 5]
    strategy_stats = strategy_stats.sort_values('syc_mean', ascending=False)

    print("\nStrategy breakdown (sorted by sycophancy, min 5 examples):")
    print(strategy_stats.to_string())

    groups = [group['sycophantic'].values
              for name, group in df.groupby('strategy') if len(group) >= 5]
    if len(groups) >= 2:
        f_stat, p_value = stats.f_oneway(*groups)
        print(f"\n  ANOVA (sycophancy ~ strategy): F={f_stat:.3f}, p={p_value:.4f}")
        if p_value < 0.05:
            print("  => Significant difference in sycophancy across strategies")

    strategy_stats.to_csv("results/by_strategy.csv")
    return strategy_stats


def by_emotion(df):
    print("\n" + "=" * 70)
    print("SYCOPHANCY BY SEEKER EMOTION")
    print("=" * 70)

    emotion_stats = df.groupby('emotion_type').agg({
        'sycophantic': ['mean', 'std', 'count'],
        'therapeutic': 'mean',
    }).round(3)
    emotion_stats.columns = ['syc_mean', 'syc_std', 'count', 'therapeutic']
    emotion_stats = emotion_stats.sort_values('syc_mean', ascending=False)
    print(emotion_stats.to_string())

    emotion_stats.to_csv("results/by_emotion.csv")
    return emotion_stats


def by_problem(df):
    print("\n" + "=" * 70)
    print("SYCOPHANCY BY PROBLEM TYPE")
    print("=" * 70)

    problem_stats = df.groupby('problem_type').agg({
        'sycophantic': ['mean', 'std', 'count'],
        'therapeutic': 'mean',
    }).round(3)
    problem_stats.columns = ['syc_mean', 'syc_std', 'count', 'therapeutic']
    problem_stats = problem_stats.sort_values('syc_mean', ascending=False)
    print(problem_stats.to_string())

    problem_stats.to_csv("results/by_problem.csv")
    return problem_stats


def correlations(df):
    print("\n" + "=" * 70)
    print("DIMENSION CORRELATIONS")
    print("=" * 70)

    dimensions = ['sycophantic', 'therapeutic', 'empathy', 'helpful', 'boundaried']
    corr_matrix = df[dimensions].corr().round(3)
    print("\nCorrelation matrix:")
    print(corr_matrix.to_string())

    print("\nKey correlations with sycophancy:")
    for dim in ['therapeutic', 'boundaried', 'helpful', 'empathy']:
        corr, p = stats.pearsonr(df['sycophantic'], df[dim])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  sycophantic × {dim:15s}: r={corr:.3f}, p={p:.4f} {sig}")


def high_vs_low_examples(df):
    print("\n" + "=" * 70)
    print("EXAMPLE RESPONSES")
    print("=" * 70)

    print("\nHIGH SYCOPHANCY (score > 0.7):")
    for _, row in df[df['sycophantic'] > 0.7].head(3).iterrows():
        print(f"\n  Seeker:    {row['seeker_msg'][:150]}...")
        print(f"  Supporter: {row['supporter_msg'][:150]}...")
        print(f"  Score: {row['sycophantic']:.2f} | Strategy: {row['strategy']}")

    print("\n\nLOW SYCOPHANCY (score < 0.3):")
    for _, row in df[df['sycophantic'] < 0.3].head(3).iterrows():
        print(f"\n  Seeker:    {row['seeker_msg'][:150]}...")
        print(f"  Supporter: {row['supporter_msg'][:150]}...")
        print(f"  Score: {row['sycophantic']:.2f} | Strategy: {row['strategy']}")


def save_summary(df, filepath="results/summary.txt"):
    syc_mean = df['sycophantic'].mean()
    with open(filepath, 'w') as f:
        f.write("ESConv Human Sycophancy Baseline — Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total responses analyzed: {len(df)}\n")
        f.write(f"Unique conversations: {df['conv_id'].nunique()}\n\n")
        f.write("Dimension means:\n")
        for dim in ['sycophantic', 'therapeutic', 'empathy', 'helpful', 'boundaried']:
            f.write(f"  {dim}: {df[dim].mean():.3f} ± {df[dim].std():.3f}\n")
        f.write(f"\nHuman sycophancy baseline: {syc_mean:.3f}\n")
    print(f"\nSummary saved to {filepath}")


def main():
    Path("results").mkdir(exist_ok=True)

    scores_df = load_results()
    conv_df = load_conv_feedback()

    overall_summary(scores_df)

    if conv_df is not None:
        satisfaction_correlation(scores_df, conv_df)

    temporal_analysis(scores_df)
    by_strategy(scores_df)
    by_emotion(scores_df)
    by_problem(scores_df)
    correlations(scores_df)
    high_vs_low_examples(scores_df)
    save_summary(scores_df)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nOutput files:")
    print("  results/conv_level_analysis.csv  (satisfaction correlation)")
    print("  results/temporal_analysis.csv    (early/middle/late arc)")
    print("  results/by_strategy.csv")
    print("  results/by_emotion.csv")
    print("  results/by_problem.csv")
    print("  results/summary.txt")


if __name__ == "__main__":
    main()
