# IMDb Sentiment Analysis

A **sentiment analysis model** trained on IMDb movie reviews to classify text as **positive** or **negative**.  
This project demonstrates **text preprocessing, Word2Vec embeddings, and Logistic Regression** in a Streamlit web app for real-time prediction.

---

## 📖 About the Project

This model was trained on the **IMDb Movie Reviews Dataset** (50,000 reviews) for sentiment classification.  

- **Dataset Information**  
  - RangeIndex: 50,000 entries (0 to 49,999)  
  - Columns (2):  
    - `review` → 50,000 non-null, object  
    - `sentiment` → 50,000 non-null, object  
  - **Label Meaning:**  
    - `positive` → 25,000 entries  
    - `negative` → 25,000 entries  
  - **Text column:** Contains the actual movie review text for classification.  
  - **Source:** [IMDb Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)  

---

## ⚡ Preprocessing Steps

1. Lowercasing  
2. Removing HTML tags  
3. Removing punctuation  
4. Removing stopwords  
5. Tokenization using **spaCy**  
6. Word2Vec vectorization  

---

## 🧠 Model Development

- **Embedding:** Word2Vec (vector size = 200, window = 10, min_count = 2)  
- **Classifier:** Logistic Regression (max_iter = 1000)  
- **Performance on Test Set:**  
  - Accuracy: 0.88696  
  - Precision: 0.89  
  - Recall: 0.88  
  - F1-Score: 0.89  

- **Saved Models:**  
  - `w2v_imdb.model` → Word2Vec embeddings  
  - `lr_imdb_model.pkl` → Trained Logistic Regression classifier  

---

## 🛠 Usage

1. **Clone the repository**  

```bash
git clone https://github.com/hasnainyaqub/IMDb_Sentiment_Analysis.git
cd IMDb_Sentiment_Analysis

