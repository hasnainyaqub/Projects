import streamlit as st
import re
import string
import gensim
import joblib
import numpy as np
from nltk.corpus import stopwords
import spacy
import subprocess

# Try to load English model; download if missing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# ------------------------
# Load Models
# ------------------------
@st.cache_resource
def load_models():
    # Load Word2Vec
    w2v_model = gensim.models.Word2Vec.load("w2v_imdb.model")
    
    # Load classifier
    lr_model = joblib.load("lr_imdb_model.pkl")
    
    # Load spaCy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    
    # Stopwords
    stop = set(stopwords.words('english'))
    
    return w2v_model, lr_model, nlp, stop

w2v_model, lr_model, nlp, stop = load_models()


# ------------------------
# Preprocessing
# ------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub("<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join([word for word in text.split() if word not in stop])
    return text

def tokenize(text):
    doc = nlp(text)
    return [token.text for token in doc]

def document_vector(tokens):
    valid_words = [w for w in tokens if w in w2v_model.wv.key_to_index]
    if not valid_words:
        return np.zeros(w2v_model.vector_size)
    return np.mean(w2v_model.wv[valid_words], axis=0)


# ------------------------
# Streamlit UI
# ------------------------
st.title("IMDB Sentiment Analysis")

user_input = st.text_area("Enter a movie review:")

if st.button("Predict Sentiment"):
    if not user_input.strip():
        st.warning("Please enter a review.")
    else:
        # Preprocess
        cleaned = preprocess_text(user_input)
        tokens = tokenize(cleaned)
        
        # Vectorize
        vector = document_vector(tokens).reshape(1, -1)
        
        # Predict
        pred = lr_model.predict(vector)[0]
        proba = lr_model.predict_proba(vector)[0]
        
        # Display
        st.write("**Predicted Sentiment:**", "Positive" if pred == 1 else "Negative")
        st.write("**Confidence:**", f"{np.max(proba)*100:.2f}%")


# ------------------------
# Sidebar - Model Information
# ------------------------
st.sidebar.title("📊 Model Information")
st.sidebar.markdown("""
**Logistic Regression Results:**

- Accuracy:  `0.88696`  
- Precision: `0.89`  
- Recall:    `0.88`  
- F1-Score:  `0.89`  
""")
st.sidebar.markdown("---")
st.sidebar.subheader("🌐 Connect with Me")
st.sidebar.write("[LinkedIn](https://www.linkedin.com/in/hasnainyaqoob/)")
st.sidebar.write("[GitHub](https://github.com/hasnainyaqub)")
st.sidebar.write("[Kaggle](https://www.kaggle.com/hasnainyaqooob)")


# Example reviews
positive_reviews = [
    "Absolutely loved this movie! Amazing acting and story.",
    "A heartwarming and emotional film. Highly recommend!",
    "Stunning visuals and brilliant soundtrack.",
    "An enjoyable film with funny moments and likable characters.",
    "A beautifully crafted story with memorable performances."
]

negative_reviews = [
    "What a waste of time. Predictable plot and boring characters.",
    "I could barely finish it. Terrible pacing and awful dialogue.",
    "Horrible movie. I don't understand why it got good reviews.",
    "Disappointing. The story had potential but poor execution.",
    "Mediocre at best. Felt like the director didn't know what they wanted."
]

# Display side by side
st.subheader("Example Reviews")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Positive Reviews:**")
    for review in positive_reviews:
        st.write(f"- {review}")

with col2:
    st.markdown("**Negative Reviews:**")
    for review in negative_reviews:
        st.write(f"- {review}")

# ------------------------
# About this Project
# ------------------------


st.subheader("📖 About this Project")
st.markdown("""
This model was trained on the [IMDB
 Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (50,000 reviews) for sentiment analysis.

---

📊 **Dataset Information**

RangeIndex: 50,000 entries (0 to 49,999)  
Columns (2):  
- `review` → 50,000 non-null, object  
- `sentiment` → 50,000 non-null, object  

Memory usage: ~20 MB (approximate, depends on system)  

**Label Meaning:**  
- `positive` → 25,000 entries  
- `negative` → 25,000 entries  

**Text column:**  
Contains the actual movie review text used for classification.
""")

# ------------------------
# Disclaimer
# ------------------------
st.markdown("""
---

### ⚠️ Disclaimer
While this IMDb Sentiment Analysis model achieves **high accuracy** on the test dataset, it is **not a production-ready model**.  
Predictions may sometimes be **incorrect**, especially on ambiguous or sarcastic reviews.  
It should **not** be relied upon for critical decision-making or automated moderation.  

This project is intended for **learning, experimentation, and demonstration purposes only**.
""")

