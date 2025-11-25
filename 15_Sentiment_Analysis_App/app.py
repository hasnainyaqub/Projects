import streamlit as st
from transformers import pipeline

# Load the sentiment analysis pipeline
sentiment_pipe = pipeline("sentiment-analysis")

def main():
    st.title("Sentiment Analyzer using Transformers")

    st.write("Enter any text and get the predicted sentiment using a pretrained Transformer model.")

    user_input = st.text_area("Your text")

    if st.button("Analyze"):
        if user_input.strip():
            result = sentiment_pipe(user_input)[0]
            label = result["label"]
            score = result["score"]

            st.subheader("Prediction")
            st.write(f"Sentiment: {label}")
            st.write(f"Confidence: {score:.4f}")
        else:
            st.warning("Please enter some text.")

if __name__ == "__main__":
    main()
