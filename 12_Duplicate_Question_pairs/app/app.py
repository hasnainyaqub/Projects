import streamlit as st
import joblib
import re
import numpy as np
from bs4 import BeautifulSoup
from scipy.sparse import hstack

# -------------------------------
# Load Model and Vectorizer
# -------------------------------
@st.cache_resource
def load_model():
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('tfidf_duplicate_model.pkl')
    return vectorizer, model

vectorizer, model = load_model()

# -------------------------------
# Preprocessing Function
# -------------------------------
def preprocess(q):
    q = str(q).lower().strip()

    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    q = q.replace('[math]', '')

    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    contractions = {
        "can't": "cannot",
        "won't": "will not",
        "i'm": "i am",
        "it's": "it is",
        "he's": "he is",
        "she's": "she is",
        "they're": "they are",
        "we're": "we are",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "couldn't": "could not",
        "shouldn't": "should not",
        "wouldn't": "would not",
        "hasn't": "has not",
        "haven't": "have not",
        "hadn't": "had not"
    }

    q_decontracted = []
    for word in q.split():
        if word in contractions:
            word = contractions[word]
        q_decontracted.append(word)
    q = ' '.join(q_decontracted)

    q = BeautifulSoup(q, "html.parser").get_text()
    q = re.sub(r'\W', ' ', q).strip()

    return q

# -------------------------------
# Feature Calculation
# -------------------------------
def calculate_word_features(q1_text, q2_text):
    w1 = set(q1_text.split())
    w2 = set(q2_text.split())
    num_common = len(w1 & w2)
    total_words = len(w1) + len(w2)
    word_share = round(num_common / total_words, 2) if total_words > 0 else 0
    return num_common, total_words, word_share

def extract_features(q1, q2):
    q1_len = len(q1)
    q2_len = len(q2)
    q1_num_words = len(q1.split())
    q2_num_words = len(q2.split())
    num_common, word_total, word_share = calculate_word_features(q1, q2)
    return np.array([[q1_len, q2_len, q1_num_words, q2_num_words, num_common, word_total, word_share]])

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Quora Question Pairs Duplicate Detector")
st.write("Enter two questions to check if they are duplicates.")

q1_input = st.text_area("Question 1")
q2_input = st.text_area("Question 2")

if st.button("Check Duplicate"):
    if q1_input and q2_input:
        q1 = preprocess(q1_input)
        q2 = preprocess(q2_input)

        q1_vec = vectorizer.transform([q1])
        q2_vec = vectorizer.transform([q2])

        X_text = hstack((q1_vec, q2_vec))
        X_num = extract_features(q1, q2)
        X_new = hstack((X_text, X_num))

        pred = model.predict(X_new)[0]

        if pred == 1:
            st.success("✅ These questions are duplicates.")
        else:
            st.error("❌ These questions are not duplicates.")
    else:
        st.warning("Please enter both questions before predicting.")

st.markdown("---")
st.caption("Model: TF-IDF + ML classifier trained on Quora Question Pairs dataset.")
