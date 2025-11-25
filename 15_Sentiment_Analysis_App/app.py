import streamlit as st
import pandas as pd
from transformers import pipeline
import os


st.set_page_config(page_title="Sentiment Analyzer", layout="wide")


sentiment_pipe = pipeline(
    task="sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

FILE_PATH = "reviews.csv"

# Load CSV or create it
def load_reviews():
    if os.path.exists(FILE_PATH):
        return pd.read_csv(FILE_PATH)
    else:
        df = pd.DataFrame(columns=["review", "sentiment"])
        df.to_csv(FILE_PATH, index=False)
        return df

def save_review(text, sentiment):
    df = load_reviews()
    df.loc[len(df)] = [text, sentiment]
    df.to_csv(FILE_PATH, index=False)

# Custom CSS
st.markdown(
    """
    <style>
        body { background-color: #f7f9fc; }
        .main { background-color: #f7f9fc; }

        .stTextArea textarea {
            border-radius: 10px;
            border: 2px solid #d0d7de;
        }

        .prediction-box {
            padding: 15px;
            background: #e8f5e9;
            border-radius: 10px;
            border: 1px solid #b2dfdb;
            font-size: 18px;
            font-weight: bold;
        }

        .review-card {
            background: white;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 12px;
            border: 1px solid #e3e6ea;
            box-shadow: 0px 2px 4px rgba(0,0,0,0.05);
        }

        .pos-header {
            color: green;
            font-size: 20px;
            font-weight: bold;
        }

        .neg-header {
            color: red;
            font-size: 20px;
            font-weight: bold;
        }

        .title-style {
            font-size: 32px;
            font-weight: bold;
            color: #34495e;
        }

        .button-style button {
            background-color: #4a90e2 !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
            border: none !important;
        }

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='title-style'>Sentiment Review App</h1>", unsafe_allow_html=True)

menu = st.sidebar.selectbox("Choose Page", ["Sentiment Checker", "Reviews Page"])

# Page 1
if menu == "Sentiment Checker":
    st.subheader("Check Sentiment")

    text = st.text_area("Write something for sentiment prediction")

    if st.button("Analyze"):
        if text.strip():
            result = sentiment_pipe(text)[0]
            label = result["label"]
            score = result["score"]

            color = "#e8f5e9" if label == "POSITIVE" else "#ffebee"

            st.markdown(
                f"""
                <div class="prediction-box" style="background:{color}">
                    Sentiment: {label}<br>
                    Confidence: {score:.4f}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("Write something first")

# Page 2
elif menu == "Reviews Page":
    st.subheader("Submit a Review")

    review_text = st.text_area("Write a review")

    if st.button("Submit Review"):
        if review_text.strip():
            result = sentiment_pipe(review_text)[0]
            sentiment = result["label"]
            save_review(review_text, sentiment)
            st.success(f"Review saved as {sentiment}")
        else:
            st.warning("Write a review first")

    df = load_reviews()

    st.subheader("All Reviews")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='pos-header'>Positive Reviews</div>", unsafe_allow_html=True)
        positive = df[df["sentiment"] == "POSITIVE"]
        for _, row in positive.iterrows():
            st.markdown(
                f"""
                <div class='review-card'>
                    {row['review']}
                </div>
                """,
                unsafe_allow_html=True
            )

    with col2:
        st.markdown("<div class='neg-header'>Negative Reviews</div>", unsafe_allow_html=True)
        negative = df[df["sentiment"] == "NEGATIVE"]
        for _, row in negative.iterrows():
            st.markdown(
                f"""
                <div class='review-card'>
                    {row['review']}
                </div>
                """,
                unsafe_allow_html=True
            )
