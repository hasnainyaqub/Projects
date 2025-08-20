from flask import Flask, render_template, request
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize
app = Flask(__name__)

# Load models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [ps.stem(word) for word in tokens]
    return " ".join(tokens)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        input_sms = request.form['message']
        transformed = transform_text(input_sms)
        vector_input = tfidf.transform([transformed])
        prediction = model.predict(vector_input)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
