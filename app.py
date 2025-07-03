import streamlit as st
import pickle

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email/SMS Spam Classifier')



import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import swifter

# Load once
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [ps.stem(word) for word in tokens]
    return " ".join(tokens)


input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1 . preprocess
    transform_email = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transform_email])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
