import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import dateparser
import re
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="Hotel Kazakhstan • 2GIS Analytics",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏨 Hotel Kazakhstan (Almaty) - 2GIS Reviews Analytics")
st.markdown("**Analysis of 1,227 reviews from 2020 to 2025**")

# ====================== DATA LOADING ======================
@st.cache_data
def load_data():
    """Load and preprocess the 2GIS reviews dataset."""
    df = pd.read_csv("2gis_kazakhstan_reviews.csv")
    
    # Parse Russian dates
    def parse_russian_date(date_str):
        if pd.isna(date_str) or str(date_str).strip() == "":
            return None
        # Remove ", edited" part
        clean_date = str(date_str).split(",")[0].strip()
        parsed = dateparser.parse(clean_date, languages=['ru'])
        return parsed
    
    df["date"] = df["date"].apply(parse_russian_date)
    df = df.dropna(subset=["date"])
    
    # Convert rating to numeric
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    
    # Create time features
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    df["year"] = df["date"].dt.year
    
    # Sentiment classification
    def get_sentiment(rating):
        if rating >= 4:
            return "Positive"
        elif rating <= 2:
            return "Negative"
        else:
            return "Neutral"
    
    df["sentiment"] = df["rating"].apply(get_sentiment)
    
    # Clean text for word cloud and frequency analysis
    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^а-яёa-z\s]', ' ', text)   # Keep only letters
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    df["clean_text"] = df["text"].apply(clean_text)
    
    return df

df = load_data()

# ====================== SIDEBAR FILTERS ======================
st.sidebar.header("🔍 Filters")

date_range = st.sidebar.date_input(
    "Date Range",
    value=(df["date"].min().date(), df["date"].max().date()),
    min_value=df["date"].min().date(),
    max_value=df["date"].max().date()
)

rating_range = st.sidebar.slider("Rating Range", 1, 5, (1, 5))

search_text = st.sidebar.text_input("Search in review text", "")

# Apply filters
mask = (
    (df["date"].dt.date >= date_range[0]) &
    (df["date"].dt.date <= date_range[1]) &
    (df["rating"] >= rating_range[0]) &
    (df["rating"] <= rating_range[1])
)

if search_text:
    mask &= df["text"].str.contains(search_text, case=False, na=False)

filtered_df = df[mask].copy()

# ====================== KEY METRICS ======================
st.subheader("📊 Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Reviews", len(filtered_df))
col2.metric("Average Rating", f"{filtered_df['rating'].mean():.2f} ⭐")
col3.metric("5-Star Reviews", f"{(filtered_df['rating'] == 5).mean():.1%}")
col4.metric("Positive", (filtered_df['sentiment'] == "Positive").sum())
col5.metric("Negative", (filtered_df['sentiment'] == "Negative").sum())

# ====================== TABS ======================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Rating Overview",
    "📅 Trends Over Time",
    "☁️ Text Analysis",
    "📋 Sample Reviews"
])

with tab1:
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Rating Distribution")
        fig_rating = px.histogram(
            filtered_df, x="rating", nbins=5,
            color_discrete_sequence=["#6366f1"],
            title="Number of Reviews by Rating"
        )
        st.plotly_chart(fig_rating, use_container_width=True)
    
    with col_b:
        st.subheader("Sentiment Distribution")
        sentiment_counts = filtered_df["sentiment"].value_counts()
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map={
                "Positive": "#22c55e",
                "Negative": "#ef4444",
                "Neutral": "#eab308"
            },
            title="Review Sentiment"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    st.subheader("Reviews and Average Rating Over Time")
    
    monthly = filtered_df.groupby("year_month").agg(
        review_count=("rating", "count"),
        avg_rating=("rating", "mean")
    ).reset_index()
    
    fig_trend = go.Figure()
    
    fig_trend.add_trace(go.Bar(
        x=monthly["year_month"],
        y=monthly["review_count"],
        name="Number of Reviews",
        marker_color="#6366f1"
    ))
    
    fig_trend.add_trace(go.Scatter(
        x=monthly["year_month"],
        y=monthly["avg_rating"],
        name="Average Rating",
        yaxis="y2",
        line=dict(color="#f59e0b", width=3)
    ))
    
    fig_trend.update_layout(
        title="Monthly Review Trends",
        yaxis=dict(title="Number of Reviews"),
        yaxis2=dict(title="Average Rating", overlaying="y", side="right", range=[1, 5]),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)

with tab3:
    st.subheader("Word Cloud from Reviews")
    
    all_text = " ".join(filtered_df["clean_text"])
    
    if all_text.strip():
        wordcloud = WordCloud(
            width=900,
            height=450,
            background_color="white",
            colormap="viridis",
            max_words=120
        ).generate(all_text)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info("No text available to generate word cloud.")

    st.subheader("Top 20 Most Frequent Words")
    words = " ".join(filtered_df["clean_text"]).split()
    word_freq = Counter(words)
    top_words = pd.DataFrame(word_freq.most_common(20), columns=["Word", "Frequency"])
    st.dataframe(top_words, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Recent Reviews")
    sample = filtered_df.sort_values("date", ascending=False).head(15)
    sample_display = sample[["date", "rating", "text", "sentiment"]].copy()
    sample_display["date"] = sample_display["date"].dt.strftime("%d %B %Y")
    
    st.dataframe(
        sample_display,
        column_config={
            "rating": st.column_config.NumberColumn("⭐ Rating"),
            "text": st.column_config.TextColumn("Review Text"),
            "sentiment": st.column_config.TextColumn("Sentiment")
        },
        use_container_width=True,
        hide_index=True
    )

# Raw data expander
with st.expander("🗃️ View Raw Filtered Data"):
    st.dataframe(filtered_df, use_container_width=True)

st.caption("2GIS Reviews Analytics Dashboard for Hotel Kazakhstan, Almaty")
