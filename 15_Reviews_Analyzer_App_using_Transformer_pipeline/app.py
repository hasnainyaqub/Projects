import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px
import time
import os

# Page config
st.set_page_config(page_title="Sentiment Analyzer", layout="wide")

# Initial CSS styling
st.markdown("""
    <style>
        .positive-box {
            background-color: #d4edda;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 8px;
        }
        .negative-box {
            background-color: #f8d7da;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 8px;
        }
        .badge-positive {
            color: white;
            background-color: #28a745;
            padding: 5px 12px;
            border-radius: 8px;
            font-weight: bold;
        }
        .badge-negative {
            color: white;
            background-color: #dc3545;
            padding: 5px 12px;
            border-radius: 8px;
            font-weight: bold;
        }
        .example-box {
            background-color: #ffffff;
            padding: 12px;
            border-radius: 10px;
            border: 1px solid #ddd;
            margin-bottom: 12px;
        }
    </style>
""", unsafe_allow_html=True)

# Theme toggle
theme = st.sidebar.radio("Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown(
        """
        <style>
            body { background-color: #1e1e1e; color: #f0f0f0; }
            .example-box { background-color: #2d2d2d; border-color: #555; }
            .positive-box { background-color: #234d33; }
            .negative-box { background-color: #4d2323; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Load model
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

# File path
FILE_PATH = "reviews.csv"

def load_reviews():
    if os.path.exists(FILE_PATH):
        return pd.read_csv(FILE_PATH)
    df = pd.DataFrame(columns=["review", "sentiment"])
    df.to_csv(FILE_PATH, index=False)
    return df

def save_review(text, sentiment):
    df = load_reviews()
    df.loc[len(df)] = [text, sentiment]
    df.to_csv(FILE_PATH, index=False)

st.title("📥  Reviews Analyzer App")

menu = st.sidebar.selectbox("Navigate", ["Sentiment Checker", "Reviews Page"])

# Sentiment Checker page
if menu == "Sentiment Checker":
    st.subheader("🎯 Sentiment Prediction")

    user_text = st.text_area("Write your text here")
    animated_placeholder = st.empty()

    if st.button("Analyze Sentiment"):
        if user_text.strip():
            animated_placeholder.text("⏳ Analyzing, please wait")
            time.sleep(1.2)

            result = sentiment_pipe(user_text)[0]
            label = result["label"]
            score = result["score"]

            animated_placeholder.empty()

            if label == "POSITIVE":
                emoji = "😊"
                st.markdown(f"<p class='badge-positive'>{emoji} Positive</p>", unsafe_allow_html=True)
            else:
                emoji = "😡"
                st.markdown(f"<p class='badge-negative'>{emoji} Negative</p>", unsafe_allow_html=True)

            st.info(f"Confidence Score: {score:.4f}")
        else:
            st.warning("Write text before analyzing")

    st.markdown("### ⭐ Example Sentences")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 👍 Positive Examples")
        for s in [
            "This product is amazing",
            "I love this so much",
            "Such a beautiful experience"
        ]:
            st.markdown(f"<div class='example-box'>✔️ {s}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("#### 👎 Negative Examples")
        for s in [
            "This was terrible",
            "Very disappointed",
            "I hate this service"
        ]:
            st.markdown(f"<div class='example-box'>❌ {s}</div>", unsafe_allow_html=True)

# Reviews Page
elif menu == "Reviews Page":
    st.subheader("📝 Submit a Review with Sentiment")

    review_text = st.text_area("Write your review")
    animated_placeholder = st.empty()

    if st.button("Submit Review"):
        if review_text.strip():
            animated_placeholder.text("Saving, please wait")
            time.sleep(1)
            result = sentiment_pipe(review_text)[0]
            sentiment = result["label"]

            save_review(review_text, sentiment)
            animated_placeholder.empty()

            if sentiment == "POSITIVE":
                st.success("😊 Positive Review Saved")
            else:
                st.error("😡 Negative Review Saved")
        else:
            st.warning("Write something before submitting")

    df = load_reviews()

    st.markdown("### 📊 Reviews Sentiment Chart")
    if not df.empty:
        chart_df = df.groupby("sentiment").size().reset_index(name="count")
        fig = px.pie(chart_df, values="count", names="sentiment", title="Sentiment Distribution")
        st.plotly_chart(fig)

    col1, col2 = st.columns(2)

    with col1:
        st.write("### 👍 Positive Reviews")
        pos = df[df["sentiment"] == "POSITIVE"]
        for _, row in pos.iterrows():
            st.markdown(f"<div class='positive-box'>✔️ {row['review']}</div>", unsafe_allow_html=True)

    with col2:
        st.write("### 👎 Negative Reviews")
        neg = df[df["sentiment"] == "NEGATIVE"]
        for _, row in neg.iterrows():
            st.markdown(f"<div class='negative-box'>❌ {row['review']}</div>", unsafe_allow_html=True)



# Footer
st.markdown(
    """
    <hr>
    <div style='text-align:center; padding:10px;'>
        <p style='font-size:16px; margin:0;'>
            Built with ❤️ by <strong>Hasnain Yaqoob</strong>
        </p>
        <p style='font-size:14px; margin:5px 0;'>
            AI Engineer | ML and NLP Enthusiast
        </p>
        <p style='margin:0;'>
            <a href='https://www.linkedin.com/in/hasnainyaqoob' target='_blank'>LinkedIn</a> |
            <a href='https://x.com/Hasnain_Yaqoob_' target='_blank'>X</a> |
            <a href='https://github.com/hasnainyaqub' target='_blank'>GitHub</a> |
            <a href='https://www.kaggle.com/hasnainyaqooob' target='_blank'>Kaggle</a> 
        </p>
        <p style='color:gray; font-size:12px; margin-top:5px;'>
            © 2025 All rights reserved
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
