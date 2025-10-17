# 📧 Spam Email Classification (Streamlit App)

This project is a **Spam Email Classifier** built using **Machine Learning** and deployed with **Streamlit**. The model predicts whether an email message is **Spam** or **Ham (legitimate)** based on text content.  

## 🚀 Features
- Interactive **Streamlit web app** for spam detection  
- Preprocessing with **NLTK** (tokenization, stopword removal, stemming)  
- TF-IDF Vectorizer for feature extraction  
- **ExtraTreesClassifier** trained on 83k+ emails  
- Model performance:  
  - Accuracy: 98.64%  
  - Precision: 98.75%  
  - Recall: 98.66%  
  - F1-Score: 98.70%  
- Example spam/ham messages included for quick testing  
- User-friendly interface with sidebar model info and links  

## 📊 Dataset
Dataset used: [Email Spam Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset)  

**Dataset details:**  
- **Entries:** 83,448 emails  
- **Columns:**  
  - `label` → (1 = Spam, 0 = Ham)  
  - `text` → actual email content  
- **Distribution:**  
  - Spam: 43,910  
  - Ham: 39,538  

## ⚠️ Disclaimer
This project is for **learning and demonstration purposes only**.  
Although the model performs well on test data, it is **not a production-ready system**.  
It may misclassify some messages, so do not use it for sensitive or critical applications.  

## 🌐 Live Demo
[Click here to try the app](https://myspamemail.streamlit.app/)  

## 🛠️ Installation
1. Clone the repo  
   ```bash
   git clone https://github.com/hasnainyaqub/Spam_email_classification.git
   cd Spam_email_classification

2. Install dependencies
   ```bash
   pip install -r requirements.txt

3. Run the app locally
   ```bash
    streamlit run app.py
   
## Connect with Me
-  [**LinkedIn**](https://www.linkedin.com/in/hasnainyaqoob/)
-  [**Kaggle**](https://www.kaggle.com/hasnainyaqooob)
- [**X.com**](https://x.com/hasnain_yaqoob_) 
- [**GitHub**](https://github.com/hasnainyaqub)
