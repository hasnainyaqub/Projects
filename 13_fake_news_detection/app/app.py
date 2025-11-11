import streamlit as st
import re
import pickle

# -------------------------------
# Load Model and Vectorizer
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -------------------------------
# Core Python Preprocessing Function
# -------------------------------
def preprocess(text):
    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove punctuation and special chars (keep spaces)
    text = re.sub(r"[^a-z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove very short words (like “a”, “an”, “of”)
    words = [word for word in text.split() if len(word) > 2]

    # Join back into one string
    clean_text = " ".join(words)
    return clean_text

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.title("📰 Fake News Detection App")
st.write("Enter a news article below to check whether it’s **Real** or **Fake**.")

# Text input box
user_input = st.text_area("Paste your news content here", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        # Preprocess the input text
        clean_text = preprocess(user_input)

        # Vectorize
        text_vector = vectorizer.transform([clean_text])

        # Predict
        prediction = model.predict(text_vector)[0]
        prob = model.predict_proba(text_vector)[0]

        # Display result
        if prediction == 1:
            st.success(f"✅ This news looks **REAL** (confidence: {prob[1]*100:.2f}%)")
        else:
            st.error(f"🚨 This news looks **FAKE** (confidence: {prob[0]*100:.2f}%)")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("**Developed by [Your Name] | Fake News Detection using Machine Learning**")
