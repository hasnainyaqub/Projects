# 🌐 AI Language Translator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://languagetranslationnapp.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modern, AI-powered language translation web application that translates English text to 22+ languages using state-of-the-art neural machine translation models.

## 🚀 Live Demo

**Try it now:** [https://languagetranslationnapp.streamlit.app/](https://languagetranslationnapp.streamlit.app/)

## ✨ Features

- 🌍 **22 Languages Supported** - Translate English to Urdu, French, Spanish, German, Italian, and many more
- 🤖 **AI-Powered** - Uses Helsinki-NLP OPUS-MT models based on Marian NMT framework
- ⚡ **Real-time Translation** - Instant translation with just one click
- 🎨 **Modern UI** - Clean, intuitive, and responsive interface
- 💡 **Quick Examples** - Pre-loaded example sentences for easy testing
- 🔄 **Smart Caching** - Models are cached for faster subsequent translations
- 📱 **Mobile Friendly** - Works seamlessly on all devices

## 🌍 Supported Languages

| Language | Flag | Language | Flag |
|----------|------|----------|------|
| Urdu | 🇵🇰 | Swedish | 🇸🇪 |
| French | 🇫🇷 | Romanian | 🇷🇴 |
| Spanish | 🇪🇸 | Hungarian | 🇭🇺 |
| German | 🇩🇪 | Finnish | 🇫🇮 |
| Italian | 🇮🇹 | Danish | 🇩🇰 |
| Portuguese | 🇵🇹 | Norwegian | 🇳🇴 |
| Russian | 🇷🇺 | Ukrainian | 🇺🇦 |
| Chinese | 🇨🇳 | Persian | 🇮🇷 |
| Japanese | 🇯🇵 | Arabic | 🇸🇦 |
| Hindi | 🇮🇳 | Turkish | 🇹🇷 |
| Dutch | 🇳🇱 | Polish | 🇵🇱 |

## 🛠️ Technologies Used

- **Streamlit** - Web application framework
- **Transformers** - Hugging Face library for NLP models
- **PyTorch** - Deep learning framework
- **Helsinki-NLP OPUS-MT** - Pre-trained translation models
- **SentencePiece** - Tokenization library

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/hasnainyaqub/ai-language-translator.git
cd ai-language-translator
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

## 📋 Requirements
```txt
streamlit==1.29.0
transformers==4.36.0
torch==2.1.0
sentencepiece==0.1.99
protobuf==3.20.3
```

## 🎯 How to Use

1. **Select Target Language** - Click on any language button at the top
2. **Enter Text** - Type or paste your English text in the input box
3. **Translate** - Click the "🚀 Translate" button
4. **View Results** - See the translation appear instantly in the output box
5. **Try Examples** - Click on pre-loaded examples for quick testing

## 🏗️ Project Structure
```
ai-language-translator/
│
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── .streamlit/
    └── config.toml       # Streamlit configuration (optional)
```

## 🔧 Configuration

The application uses the following model configuration:
```python
LANGUAGES = {
    "🇵🇰 Urdu": {"model": "Helsinki-NLP/opus-mt-en-ur", "code": "ur"},
    "🇫🇷 French": {"model": "Helsinki-NLP/opus-mt-en-fr", "code": "fr"},
    "🇪🇸 Spanish": {"model": "Helsinki-NLP/opus-mt-en-es", "code": "es"},
    "🇩🇪 German": {"model": "Helsinki-NLP/opus-mt-en-de", "code": "de"},
    "🇮🇹 Italian": {"model": "Helsinki-NLP/opus-mt-en-it", "code": "it"},
    "🇵🇹 Portuguese": {"model": "Helsinki-NLP/opus-mt-en-pt", "code": "pt"},
    "🇷🇺 Russian": {"model": "Helsinki-NLP/opus-mt-en-ru", "code": "ru"},
    "🇨🇳 Chinese": {"model": "Helsinki-NLP/opus-mt-en-zh", "code": "zh"},
    "🇯🇵 Japanese": {"model": "Helsinki-NLP/opus-mt-en-jap", "code": "ja"},
    "🇸🇦 Arabic": {"model": "Helsinki-NLP/opus-mt-en-ar", "code": "ar"},
    "🇮🇳 Hindi": {"model": "Helsinki-NLP/opus-mt-en-hi", "code": "hi"},
    "🇹🇷 Turkish": {"model": "Helsinki-NLP/opus-mt-en-tr", "code": "tr"},
    "🇳🇱 Dutch": {"model": "Helsinki-NLP/opus-mt-en-nl", "code": "nl"},
    "🇵🇱 Polish": {"model": "Helsinki-NLP/opus-mt-en-pl", "code": "pl"},
    "🇸🇪 Swedish": {"model": "Helsinki-NLP/opus-mt-en-sv", "code": "sv"},
    "🇷🇴 Romanian": {"model": "Helsinki-NLP/opus-mt-en-ro", "code": "ro"},
    "🇭🇺 Hungarian": {"model": "Helsinki-NLP/opus-mt-en-hu", "code": "hu"},
    "🇫🇮 Finnish": {"model": "Helsinki-NLP/opus-mt-en-fi", "code": "fi"},
    "🇩🇰 Danish": {"model": "Helsinki-NLP/opus-mt-en-da", "code": "da"},
    "🇳🇴 Norwegian": {"model": "Helsinki-NLP/opus-mt-en-no", "code": "no"},
    "🇺🇦 Ukrainian": {"model": "Helsinki-NLP/opus-mt-en-uk", "code": "uk"},
    "🇮🇷 Persian": {"model": "Helsinki-NLP/opus-mt-en-fa", "code": "fa"}
}
```

## 🚀 Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Deploy your app by selecting your repository
5. Streamlit Cloud will automatically handle dependencies and deployment

### Deploy to Other Platforms

- **Heroku**: Use the included `requirements.txt` and create a `Procfile`
- **AWS/GCP**: Deploy as a containerized application using Docker
- **Azure**: Use Azure App Service with Python runtime

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Helsinki-NLP](https://github.com/Helsinki-NLP) for the OPUS-MT translation models
- [Hugging Face](https://huggingface.co/) for the Transformers library
- [Streamlit](https://streamlit.io/) for the amazing web framework

## 👨‍💻 Developer

**Hasnain Yaqoob**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hasnainyaqoob)
[![Twitter](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/Hasnain_Yaqoob_)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/hasnainyaqub)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/hasnainyaqooob)

## 📧 Contact

For any queries or suggestions, feel free to reach out through the social links above.

## 🔮 Future Enhancements

- [ ] Add bidirectional translation (target language → English)
- [ ] Support for document translation (PDF, DOCX)
- [ ] Audio translation (speech-to-text → translate → text-to-speech)
- [ ] Translation history and saved translations
- [ ] API endpoint for programmatic access
- [ ] Offline mode with downloadable models
- [ ] Multi-language detection

---

⭐ If you find this project useful, please consider giving it a star!

**Made with ❤️ by Hasnain Yaqoob**