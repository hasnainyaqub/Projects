import streamlit as st
import re
from joblib import load

# -------------------------------
# Load Model and Vectorizer
# -------------------------------
model = load("logistic_regression_model.joblib")
vectorizer = load("tfidf_vectorizer.joblib")

# -------------------------------
# Core Python Preprocessing Function
# -------------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [word for word in text.split() if len(word) > 2]
    return " ".join(words)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# -------------------------------
# Header Section
# -------------------------------
st.markdown(
    """
    <div style='background-color:#4B9CD3; padding:20px; border-radius:10px'>
        <h1 style='color:white; text-align:center;'>📰 Fake News Detection App</h1>
        <p style='color:white; text-align:center;'>Paste a news article below to check whether it’s <b>Real</b> or <b>Fake</b>.</p>
    </div>
    """, unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------
# Prediction Section
# -------------------------------
st.subheader("📝 Enter News Article")

user_input = st.text_area("", height=180, placeholder="Paste your news content here...")

if st.button("Predict", key="predict_btn"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text first.")
    else:
        clean_text = preprocess(user_input)
        text_vector = vectorizer.transform([clean_text])
        prediction = model.predict(text_vector)[0]
        prob = model.predict_proba(text_vector)[0]

        if prediction == 1:
            st.success(f"✅ This news looks **REAL** (confidence: {prob[1]*100:.2f}%)")
        elif prediction == 0:
            st.error(f"🚨 This news looks **FAKE** (confidence: {prob[0]*100:.2f}%)")
        else:
            st.info("ℹ️ Unable to determine the authenticity of the news.")

st.markdown("<hr>", unsafe_allow_html=True)

# -------------------------------
# Sample News Section
# -------------------------------
st.subheader("📚 Sample News Articles")

col1, col2 = st.columns(2)

# Real News Examples
real_news_1 = "The Ministry of Education issued an official announcement confirming that the nationwide school improvement program has entered its final implementation phase. More than five thousand public schools completed their infrastructure upgrades during the previous quarter, including new classrooms, repaired roofs, and updated learning materials. The upgrades were verified through district level monitoring teams that submitted weekly progress reports to the central office. Independent evaluators from universities also reviewed documentation to ensure compliance with established guidelines. Officials noted that the program emphasizes transparent reporting along with community involvement in every district. Workshops will cover curriculum updates, assessment methods, and classroom management practices designed to improve learning outcomes. The ministry encouraged citizens to follow official updates and avoid relying on unverified claims circulating online."
real_news_2 = "The National Bureau of Statistics released new data showing steady growth in the manufacturing sector during the last fiscal quarter. Overall production increased by nearly four percent with strong performance in the textile, pharmaceuticals, and electronics industries. Analysts tracked output and supply chain performance using surveys submitted by registered manufacturing units across the country. The bureau collaborated with researchers from public universities to verify the data and ensure that the sampling process met statistical standards. The growth supported national employment and contributed positively to export activity during a period of global market uncertainty. Officials stated that tax incentives and improved access to raw materials played an important role in strengthening industrial output and ensuring long term stability."

# Fake News Examples
fake_news_1 = "A widely shared online post claimed that a hidden underground laboratory discovered a plant based formula that can cure every known infection without any scientific testing or approval. The post stated that a private group of scientists kept the formula secret because major companies were planning to suppress it. The claim also suggested that the formula was already distributed in remote villages where people experienced miraculous recoveries without hospitals. No evidence, documents, or verified sources were provided to support any of these statements. Authorities and medical experts clarified that no such discovery exists and warned the public against believing misleading viral posts created to attract attention."
fake_news_2 = "A circulating article on social media alleged that a retired engineer built a machine in his backyard that can generate unlimited electricity without any fuel or technical maintenance. The article claimed that the device could power an entire district and that government officials were secretly trying to take it away to prevent public access. The story included pictures unrelated to the claim and no confirmation from any local authority. Energy experts explained that the described technology is impossible based on known scientific principles. The claim was identified as fabricated content created to mislead readers for online engagement."

with col1:
    st.markdown(f"<div style='background-color:#DFF2BF; padding:15px; border-radius:10px'><h4>✅ Real News 1</h4><p>{real_news_1}</p></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='background-color:#DFF2BF; padding:15px; border-radius:10px; margin-top:10px;'><h4>✅ Real News 2</h4><p>{real_news_2}</p></div>", unsafe_allow_html=True)

with col2:
    st.markdown(f"<div style='background-color:#FFBABA; padding:15px; border-radius:10px'><h4>🚨 Fake News 1</h4><p>{fake_news_1}</p></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='background-color:#FFBABA; padding:15px; border-radius:10px; margin-top:10px;'><h4>🚨 Fake News 2</h4><p>{fake_news_2}</p></div>", unsafe_allow_html=True)

# -------------------------------
# Sidebar Section
# -------------------------------
st.sidebar.markdown(
    """
    📊 **Model Information**  
    **Logistic Regression Results:**  
    - Accuracy: 0.9866  
    - Precision: 0.99  
    - Recall: 0.99  
    - F1-Score: 0.99  

    🌐 **Connect with Me**  
    [LinkedIn](https://www.linkedin.com/in/hasnainyaqoob)  
    [GitHub](https://github.com/hasnainyaqub)  
    [Kaggle](https://www.kaggle.com/hasnainyaqooob)
    """
)

# -------------------------------
# Footer Section
# -------------------------------
st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("""
ℹ️ **About this Project**  
This app predicts whether a news article is real or fake using a model trained on the [Fake and Real News Dataset from Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).

📊 **Dataset Information**
            
**Label meaning:**  
0 → Fake news (23,481 entries)  
1 → Real news (21,417 entries)  
Memory usage: ~1.7 MB  

**Subject column:**  
Includes politicsNews (11272), worldnews (10145), News(9050), politics(6841), left news (4459), Government News (1570), US News (783), Middle east (778)
            


⚠️ **Disclaimer**  
While the model performs well on test data, it is not a production-grade model.  
It may occasionally misclassify news and should not be used for critical decisions or reporting.  

This project is for learning and demonstration purposes only.
""")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'> Developed by Hasnain Yaqoob | Fake News Detection using Machine Learning </p>", unsafe_allow_html=True)

