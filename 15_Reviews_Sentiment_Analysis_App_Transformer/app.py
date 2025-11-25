import streamlit as st
import pandas as pd
from transformers import pipeline
import os

sentiment_pipe = pipeline( task="sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    framework="pt")
FILE_PATH = "reviews.csv"

# Load previous reviews or create empty file
def load_reviews():
    if os.path.exists(FILE_PATH):
        return pd.read_csv(FILE_PATH)
    else:
        df = pd.DataFrame(columns=["review", "sentiment"])
        df.to_csv(FILE_PATH, index=False)
        return df

# Save a new review
def save_review(review_text, sentiment):
    df = load_reviews()
    df.loc[len(df)] = [review_text, sentiment]
    df.to_csv(FILE_PATH, index=False)

st.title("Review Sentiment Analyzer")

menu = st.sidebar.selectbox("Page", ["Sentiment Checker", "Reviews Page"])

if menu == "Sentiment Checker":
    st.subheader("Check Sentiment")
    text = st.text_area("Write your text here")

    if st.button("Analyze"):
        if text.strip():
            result = sentiment_pipe(text)[0]
            st.success(f"Sentiment: {result['label']}")
            st.info(f"Confidence: {result['score']:.4f}")
        else:
            st.warning("Enter some text first")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'> Developed by Hasnain Yaqoob | Sentiment Analysis App using Transformers </p>", unsafe_allow_html=True)

elif menu == "Reviews Page":
    st.subheader("Submit a Review")

    review_text = st.text_area("Write your review")

    if st.button("Submit Review"):
        if review_text.strip():
            result = sentiment_pipe(review_text)[0]
            sentiment = result["label"]
            save_review(review_text, sentiment)
            st.success(f"Review saved with sentiment: {sentiment}")
        else:
            st.warning("Write a review before submitting")

    st.write("")
    st.subheader("All Reviews")

    df = load_reviews()

    col1, col2 = st.columns(2)

    with col1:
        st.write("Positive Reviews")
        positive = df[df["sentiment"] == "POSITIVE"]
        for _, row in positive.iterrows():
            st.write(f"[+] {row['review']}")

    with col2:
        st.write("Negative Reviews")
        negative = df[df["sentiment"] == "NEGATIVE"]
        for _, row in negative.iterrows():
            st.write(f"[-] {row['review']}")


    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'> Developed by Hasnain Yaqoob | Sentiment Analysis On Reviews </p>", unsafe_allow_html=True)        

# sidebar in footer
st.sidebar.markdown(
    """

    🌐 **Connect with Me**  
    [LinkedIn](https://www.linkedin.com/in/hasnainyaqoob)  
    [GitHub](https://github.com/hasnainyaqub)  
    [Kaggle](https://www.kaggle.com/hasnainyaqooob)
    """
)

