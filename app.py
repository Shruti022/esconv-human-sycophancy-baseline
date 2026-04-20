"""
ESConv Sycophancy Dashboard
============================
Visualize human supporter sycophancy baseline.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from scipy import stats

st.set_page_config(
    page_title="Human Sycophancy Baseline",
    layout="wide"
)

st.title("Human Supporter Sycophancy Baseline")
st.markdown("""
**Dataset:** ESConv (Emotional Support Conversations, 1,300 real human-human conversations)
**Goal:** Establish how sycophantic trained human supporters are — baseline for AI comparison
""")


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_scores():
    p = "results/human_sycophancy_scores.csv"
    if Path(p).exists():
        return pd.read_csv(p)
    return None


@st.cache_data
def load_conv_feedback():
    p = "results/conv_feedback.csv"
    if Path(p).exists():
        df = pd.read_csv(p)
        return df if df['feedback_overall'].notna().sum() > 0 else None
    return None


@st.cache_data
def load_conv_analysis():
    p = "results/conv_level_analysis.csv"
    if Path(p).exists():
        return pd.read_csv(p)
    return None


df = load_scores()

if df is None:
    st.error("No results found. Run scoring first:")
    st.code("python score_esconv.py --project your-project-id --sample 500")
    st.stop()

conv_df = load_conv_feedback()
conv_analysis = load_conv_analysis()

# =============================================================================
# TOP METRICS
# =============================================================================

st.markdown("---")
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Turns Scored", len(df))
col2.metric("Conversations", df['conv_id'].nunique())
col3.metric("Avg Sycophancy", f"{df['sycophantic'].mean():.2f}")
col4.metric("Avg Therapeutic", f"{df['therapeutic'].mean():.2f}")
col5.metric("Avg Empathy", f"{df['empathy'].mean():.2f}")
col6.metric("Avg Boundaried", f"{df['boundaried'].mean():.2f}")

# Key finding banner
syc_mean = df['sycophantic'].mean()
if syc_mean < 0.3:
    color, finding = "#d4edda", "LOW — human supporters maintain appropriate limits"
elif syc_mean < 0.5:
    color, finding = "#fff3cd", "MODERATE — some validation bias present"
else:
    color, finding = "#f8d7da", "HIGH — significant over-validation detected"

st.markdown(f"""
<div style="background:{color}; padding:16px; border-radius:8px; margin:12px 0">
  <strong>Human sycophancy baseline = {syc_mean:.2f}</strong> &nbsp;|&nbsp; {finding}
</div>
""", unsafe_allow_html=True)

# =============================================================================
# TABS
# =============================================================================

tab_labels = ["Distributions", "Satisfaction Correlation", "Temporal Arc",
              "By Strategy", "By Emotion", "Correlations", "Examples"]
tabs = st.tabs(tab_labels)


# --- Tab 1: Distributions ---
with tabs[0]:
    st.subheader("Score Distributions")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x='sycophantic', nbins=25,
                           title='Sycophancy Score Distribution',
                           labels={'sycophantic': 'Sycophancy (0=boundaried, 1=over-validating)'})
        fig.add_vline(x=syc_mean, line_dash="dash",
                      annotation_text=f"Mean: {syc_mean:.2f}")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(df, x='therapeutic', nbins=25,
                           title='Therapeutic Technique Distribution',
                           labels={'therapeutic': 'Therapeutic Score'})
        fig.add_vline(x=df['therapeutic'].mean(), line_dash="dash",
                      annotation_text=f"Mean: {df['therapeutic'].mean():.2f}")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Overall Response Profile")
    dims = ['sycophantic', 'therapeutic', 'empathy', 'helpful', 'boundaried']
    means = [df[d].mean() for d in dims]
    fig = go.Figure(go.Scatterpolar(
        r=means + [means[0]],
        theta=dims + [dims[0]],
        fill='toself', name='Human Supporters'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    st.plotly_chart(fig, use_container_width=True)


# --- Tab 2: Satisfaction Correlation (KEY FINDING) ---
with tabs[1]:
    st.subheader("Satisfaction Correlation — The Key Finding")
    st.markdown("""
**Research question:** Do users rate sycophantic responses as *more* helpful?
If yes: the reward signal for sycophancy exists in human data, explaining why RLHF trains AI to over-validate.
""")

    if conv_analysis is not None:
        fb_cols = [c for c in ['feedback_overall', 'feedback_empathy', 'feedback_relevance',
                                'feedback_emotion_delta']
                   if c in conv_analysis.columns and conv_analysis[c].notna().sum() >= 5]

        if fb_cols:
            # Scatter: sycophancy vs overall satisfaction
            if 'feedback_overall' in fb_cols:
                fig = px.scatter(
                    conv_analysis, x='sycophantic', y='feedback_overall',
                    trendline='ols', trendline_color_override='red',
                    title='Conversation-level Sycophancy vs User Satisfaction',
                    labels={'sycophantic': 'Avg Sycophancy Score',
                            'feedback_overall': 'User Satisfaction Rating'},
                    hover_data=['conv_id']
                )
                r, p = stats.pearsonr(
                    conv_analysis['sycophantic'].dropna(),
                    conv_analysis['feedback_overall'].dropna()
                )
                fig.add_annotation(
                    x=0.05, y=0.95, xref='paper', yref='paper',
                    text=f"r = {r:.3f}, p = {p:.4f}",
                    showarrow=False, font=dict(size=14)
                )
                st.plotly_chart(fig, use_container_width=True)

            # All correlations heatmap
            score_dims = ['sycophantic', 'therapeutic', 'empathy', 'helpful', 'boundaried']
            corr_data = []
            for sd in score_dims:
                row = []
                for fd in fb_cols:
                    sub = conv_analysis[[sd, fd]].dropna()
                    r = stats.pearsonr(sub[sd], sub[fd])[0] if len(sub) >= 5 else np.nan
                    row.append(r)
                corr_data.append(row)

            corr_df = pd.DataFrame(corr_data, index=score_dims, columns=fb_cols)
            fig = px.imshow(corr_df, text_auto='.2f', color_continuous_scale='RdBu_r',
                            zmin=-1, zmax=1,
                            title='Score Dimensions vs User Satisfaction Ratings (Pearson r)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Satisfaction ratings found but too few valid values for plotting.")
    else:
        st.info("""
**No post-conversation satisfaction ratings were found in this version of the ESConv dataset.**

This analysis would run automatically if the dataset includes user feedback scores.

**Alternative:** Your team could:
1. Use ESConv's turn-level strategy labels as a proxy for "what the supporter chose to do"
2. Compare against the Kaggle/HuggingFace AI datasets your teammates are scoring
3. The *cross-dataset comparison* (human baseline vs AI scores) is still a publishable finding
""")


# --- Tab 3: Temporal Arc ---
with tabs[2]:
    st.subheader("Temporal Arc — Does Sycophancy Change Across a Conversation?")
    st.markdown("Do supporters become more validating as rapport builds?")

    if 'turn_position' in df.columns:
        df_temp = df.copy()
        df_temp['phase'] = pd.cut(df_temp['turn_position'], bins=[0, 0.33, 0.67, 1.0],
                                   labels=['Early', 'Middle', 'Late'], include_lowest=True)

        col1, col2 = st.columns(2)
        with col1:
            phase_means = df_temp.groupby('phase', observed=True)[
                ['sycophantic', 'therapeutic', 'boundaried']
            ].mean().reset_index()
            fig = px.bar(phase_means, x='phase', y='sycophantic',
                         color='sycophantic', color_continuous_scale='RdYlGn_r',
                         title='Avg Sycophancy by Conversation Phase',
                         labels={'sycophantic': 'Avg Sycophancy', 'phase': 'Conversation Phase'})
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(df_temp, x='phase', y='sycophantic',
                          title='Sycophancy Distribution by Phase',
                          labels={'sycophantic': 'Sycophancy', 'phase': 'Phase'})
            st.plotly_chart(fig, use_container_width=True)

        # Scatter with smoothed trend
        fig = px.scatter(
            df_temp.sample(min(500, len(df_temp)), random_state=42),
            x='turn_position', y='sycophantic',
            trendline='lowess', opacity=0.3,
            title='Sycophancy Score vs Turn Position (LOWESS trend)',
            labels={'turn_position': 'Turn Position (0=first, 1=last)',
                    'sycophantic': 'Sycophancy Score'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Re-run scoring to capture turn position data.")


# --- Tab 4: By Strategy ---
with tabs[3]:
    st.subheader("Sycophancy by Support Strategy")

    strategy_stats = df.groupby('strategy')['sycophantic'].agg(['mean', 'count']).reset_index()
    strategy_stats = strategy_stats[strategy_stats['count'] >= 5].sort_values('mean', ascending=False)

    fig = px.bar(strategy_stats, x='strategy', y='mean',
                 color='mean', color_continuous_scale='RdYlGn_r',
                 title='Avg Sycophancy by Strategy',
                 labels={'mean': 'Avg Sycophancy', 'strategy': 'Support Strategy'})
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(tickangle=30)
    st.plotly_chart(fig, use_container_width=True)

    # Multi-dim comparison
    multi = df.groupby('strategy')[
        ['sycophantic', 'therapeutic', 'empathy', 'helpful', 'boundaried']
    ].mean()
    fig = px.imshow(multi, text_auto='.2f', color_continuous_scale='RdYlGn',
                    title='All Dimensions by Strategy')
    st.plotly_chart(fig, use_container_width=True)


# --- Tab 5: By Emotion ---
with tabs[4]:
    st.subheader("Sycophancy by Seeker Emotion")

    emotion_stats = df.groupby('emotion_type')['sycophantic'].agg(['mean', 'count']).reset_index()
    emotion_stats = emotion_stats[emotion_stats['count'] >= 5].sort_values('mean', ascending=False)

    fig = px.bar(emotion_stats, x='emotion_type', y='mean',
                 color='mean', color_continuous_scale='RdYlGn_r',
                 title='Avg Sycophancy by Seeker Emotion')
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    pivot = df.groupby('emotion_type')[
        ['sycophantic', 'therapeutic', 'empathy', 'helpful', 'boundaried']
    ].mean()
    fig = px.imshow(pivot, text_auto='.2f', color_continuous_scale='RdYlGn',
                    title='All Dimensions by Seeker Emotion')
    st.plotly_chart(fig, use_container_width=True)


# --- Tab 6: Correlations ---
with tabs[5]:
    st.subheader("Dimension Correlations")

    dims = ['sycophantic', 'therapeutic', 'empathy', 'helpful', 'boundaried']
    corr = df[dims].corr()
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1, title='Correlation Matrix')
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(df.sample(min(300, len(df)), random_state=0),
                          x='sycophantic', y='therapeutic', trendline='ols',
                          title='Sycophancy vs Therapeutic Technique')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(df.sample(min(300, len(df)), random_state=0),
                          x='sycophantic', y='boundaried', trendline='ols',
                          title='Sycophancy vs Boundaries')
        st.plotly_chart(fig, use_container_width=True)


# --- Tab 7: Examples ---
with tabs[6]:
    st.subheader("Example Responses")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Low Sycophancy — Good Boundaries**")
        for _, row in df[df['sycophantic'] < 0.3].head(4).iterrows():
            with st.expander(f"Score: {row['sycophantic']:.2f} | {row.get('strategy', '')}"):
                st.markdown(f"**Seeker:** {row['seeker_msg']}")
                st.markdown(f"**Supporter:** {row['supporter_msg']}")
                if 'reasoning' in row:
                    st.caption(f"Reasoning: {row['reasoning']}")

    with col2:
        st.markdown("**High Sycophancy — Over-Validation**")
        for _, row in df[df['sycophantic'] > 0.7].head(4).iterrows():
            with st.expander(f"Score: {row['sycophantic']:.2f} | {row.get('strategy', '')}"):
                st.markdown(f"**Seeker:** {row['seeker_msg']}")
                st.markdown(f"**Supporter:** {row['supporter_msg']}")
                if 'reasoning' in row:
                    st.caption(f"Reasoning: {row['reasoning']}")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
**Implications for AI comparison:**
- AI score >> human baseline → AI is more sycophantic than trained humans
- AI score ≈ human baseline → AI mimics human supporter behavior
- AI score << human baseline → AI may be too challenging/cold

*Data: ESConv (thu-coai/esconv) | Scores: Gemini via Vertex AI*
""")

csv = df.to_csv(index=False)
st.download_button("Download Full Results", csv, "human_sycophancy_scores.csv", "text/csv")
