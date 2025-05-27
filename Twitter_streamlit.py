# Twitter Sentiment Analysis Streamlit App

import pandas as pd
import re
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS
from io import BytesIO
import datetime

# Set up Streamlit page
st.set_page_config(page_title="Twitter Sentiment Analysis with VADER", layout="wide")

# Title and Description
st.title("Twitter Sentiment Analysis with VADER")
st.markdown("""
Upload a CSV file containing a **'text'** or **'tweet'** column. This app analyzes sentiment using VADER, creates a WordCloud, and shows charts.
Supports emojis and hourly sentiment trends.
""")

# Preprocessing functions
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@[\w]+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text

def handle_emojis(text):
    return emoji.demojize(text)

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)['compound']

def classify_sentiment(score):
    if score >= 0.8:
        return 'Very Positive'
    elif score >= 0.05:
        return 'Positive'
    elif score <= -0.8:
        return 'Very Negative'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def generate_wordcloud(texts, custom_stopwords):
    all_text = ' '.join(texts)
    stopwords = STOPWORDS.union(set(custom_stopwords.lower().split()))
    wordcloud = WordCloud(
        width=800,
        height=600,
        background_color='white',
        stopwords=stopwords,
        colormap='coolwarm',
        max_words=200
    ).generate(all_text)
    return wordcloud

def visualize_sentiment_trends(df):
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        df['hour'] = df['timestamp'].dt.hour
        sentiment_trends = df.groupby('hour')['sentiment'].value_counts().unstack().fillna(0)
        fig, ax = plt.subplots(figsize=(10, 6))
        sentiment_trends.plot(kind='line', marker='o', ax=ax)
        ax.set_title("Sentiment Trends by Hour")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Tweet Count")
        plt.tight_layout()
        return fig
    return None

# UI Inputs
uploaded_file = st.file_uploader("Upload CSV File (must contain a 'text' column)", type=['csv'])
filter_words = st.text_input("Filter Words (separate by space)", "covid twitter vaccine")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        if 'text' not in df.columns:
            st.error("The CSV must contain a 'text' or 'tweet' column.")
        else:
            st.info("Processing... This may take a moment.")
            df['cleaned_text'] = df['text'].astype(str).apply(lambda x: preprocess_text(handle_emojis(x)))

            sentiments = []
            sentiment_counts = {"Very Positive": 0, "Positive": 0, "Neutral": 0, "Negative": 0, "Very Negative": 0}

            for idx, row in df.iterrows():
                score = analyze_sentiment(row['cleaned_text'])
                sentiment = classify_sentiment(score)
                sentiments.append(sentiment)
                sentiment_counts[sentiment] += 1

            df['sentiment'] = sentiments
            df['sentiment_score'] = df['cleaned_text'].apply(analyze_sentiment)

            st.success("Sentiment Analysis Completed!")

            # WordCloud
            st.subheader("Word Cloud")
            wordcloud = generate_wordcloud(df['cleaned_text'], filter_words)
            fig_wc, ax_wc = plt.subplots(figsize=(10, 6))
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)

            # Bar Chart
            st.subheader("Sentiment Distribution - Bar Chart")
            sentiment_counts_plot = df['sentiment'].value_counts()
            fig_bar, ax_bar = plt.subplots()
            sns.barplot(x=sentiment_counts_plot.index, y=sentiment_counts_plot.values, palette='coolwarm', ax=ax_bar)
            ax_bar.set_title('Sentiment Distribution')
            ax_bar.set_xlabel('Sentiment')
            ax_bar.set_ylabel('Count')
            st.pyplot(fig_bar)

            # Pie Chart
            st.subheader("Sentiment Distribution - Pie Chart")
            fig_pie, ax_pie = plt.subplots()
            sentiment_counts_plot.plot.pie(
                autopct='%1.1f%%',
                colors=sns.color_palette('coolwarm', n_colors=len(sentiment_counts_plot)),
                startangle=90,
                ax=ax_pie
            )
            ax_pie.set_ylabel('')
            st.pyplot(fig_pie)

            # Trends
            trend_plot = visualize_sentiment_trends(df)
            if trend_plot:
                st.subheader("Sentiment Trend by Hour")
                st.pyplot(trend_plot)

            # Data Preview
            st.subheader("Sample of Results")
            st.dataframe(df.head())

            # Sentiment Count JSON
            st.subheader("Sentiment Counts (JSON Format)")
            st.json(sentiment_counts)

            # Download CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Processed CSV",
                data=csv,
                file_name='processed_sentiment.csv',
                mime='text/csv'
            )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
