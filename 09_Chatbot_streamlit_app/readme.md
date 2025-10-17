# 🤖 Groq Chatbot (Streamlit)

A simple chatbot built with [Streamlit](https://streamlit.io/) and the [Groq API](https://console.groq.com/).  
It supports **real-time streaming responses**, a clean chat interface, and a **sidebar for API settings**.  

---

## Features
- Modern chat-style interface (messages appear in order)
- Real-time **streaming replies**
- Sidebar options:
  - Enter your **Groq API key**
  - Change API URL (default: `https://api.groq.com/openai/v1`)
  - Choose model (default: `llama-3.1-8b-instant`)
- Chat history during session

---
[Click here for Live Demo](https://chattbbot.streamlit.app/)
---
## 🛠 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/groq-chatbot.git
   cd groq-chatbot

2. **Create virtual environment and install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use venv\Scripts\activate
   pip install -r requirements.txt

3. **Add your Groq API key**
    - Get a key from [Groq Console](https://console.groq.com/keys)
    - Either set it as an environment variable:
    ```bash
     export GROQ_API_KEY="your_api_key_here"
    # or paste it in the sidebar of the app.
---
###  Run the App
    
      streamlit run main.py

  Then open http://localhost:8501 in your browser.

---
### Project Structure
    ```bash
    .
    ├── main.py          # Streamlit chatbot app
    ├── requirements.txt # Python dependencies
    └── README.md        # Project documentation


### Requirements
Create a `requirements.txt` with:
    ```bash
      streamlit
      requests

---
### Notes
- Default model: `llama-3.1-8b-instant`
- You can switch models anytime via sidebar
- This is a **minimal reference implementation**. You can extend it with:
   - "Clear Chat" button

 ----
### License

MIT License – free to use and modify.


---

Do you also want me to generate a **requirements.txt file** alongside this README so you can push both to GitHub?
