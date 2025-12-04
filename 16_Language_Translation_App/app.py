import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import torch

# Set page config
st.set_page_config(
    page_title="AI Language Translator | Hasnain Yaqoob",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for stable, modern UI
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    /* Main app styling */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        color: #1e293b;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .gradient-text {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .sub-header {
        text-align: center;
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Section headers */
    h3 {
        color: #1e293b !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Language selection section */
    .lang-section {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        height: 42px;
        border: 2px solid #e2e8f0;
        transition: all 0.2s ease;
        background-color: white;
        color: #475569;
    }
    
    .stButton > button:hover {
        border-color: #8b5cf6;
        color: #8b5cf6;
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.15);
        transform: translateY(-1px);
    }
    
    /* Primary button (selected language) */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        box-shadow: 0 8px 16px rgba(139, 92, 246, 0.3);
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 1rem;
        font-size: 1rem;
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc;
        color: #1e293b;
        transition: all 0.2s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #8b5cf6;
        background-color: white;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1);
    }
    
    /* Info/Alert boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
        background-color: #f0f9ff;
        color: #0c4a6e;
        padding: 1rem;
    }
    
    div[data-baseweb="notification"] {
        border-radius: 12px;
    }
    
    /* Success message */
    .stSuccess {
        background-color: #f0fdf4;
        color: #166534;
        border-left: 4px solid #22c55e;
    }
    
    /* Warning message */
    .stWarning {
        background-color: #fffbeb;
        color: #92400e;
        border-left: 4px solid #f59e0b;
    }
    
    /* Error message */
    .stError {
        background-color: #fef2f2;
        color: #991b1b;
        border-left: 4px solid #ef4444;
    }
    
    /* Section divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
        margin: 2rem 0;
    }
    
    /* Translation cards */
    .translation-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 2px solid #f1f5f9;
    }
    
    /* Footer styling */
    .footer {
        background: white;
        border-top: 1px solid #e2e8f0;
        padding: 1.5rem;
        text-align: center;
        margin-top: 3rem;
    }
    
    .developer-name {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.75rem;
    }
    
    .social-links {
        margin: 0.75rem 0;
    }
    
    .social-links a {
        color: #6366f1;
        text-decoration: none;
        margin: 0 12px;
        font-weight: 600;
        transition: color 0.2s ease;
    }
    
    .social-links a:hover {
        color: #8b5cf6;
        text-decoration: underline;
    }
    
    .footer-text {
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    section[data-testid="stSidebar"] h2 {
        color: #1e293b !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
    }
    
    /* Remove extra spacing */
    .element-container {
        margin-bottom: 0.5rem;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #8b5cf6 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Language configurations
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

EXAMPLE_SENTENCES = [
    "Hello, how are you?",
    "Thank you very much",
    "What is your name?",
    "Good morning",
    "I am learning a new language"
]

@st.cache_resource
def load_model(model_name):
    """Load the translation model and tokenizer"""
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return tokenizer, model, None
    except Exception as e:
        return None, None, str(e)

def translate_text(text, tokenizer, model):
    """Translate text using the loaded model"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# Initialize session state
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = "🇵🇰 Urdu"

# Header
st.markdown("<h1 class='main-header'><span class='gradient-text'>🌐 AI Language Translator</span></h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Translate English to 20+ languages using cutting-edge AI technology</p>", unsafe_allow_html=True)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Language Selection
st.markdown("### 🎯 Select Your Target Language")
st.write("")

languages_list = list(LANGUAGES.keys())
cols_per_row = 6
num_rows = (len(languages_list) + cols_per_row - 1) // cols_per_row

for row in range(num_rows):
    cols = st.columns(cols_per_row)
    for col_idx in range(cols_per_row):
        lang_idx = row * cols_per_row + col_idx
        if lang_idx < len(languages_list):
            lang = languages_list[lang_idx]
            with cols[col_idx]:
                if st.button(
                    lang, 
                    key=f"lang_{lang}",
                    use_container_width=True,
                    type="primary" if st.session_state.selected_language == lang else "secondary"
                ):
                    st.session_state.selected_language = lang
                    st.rerun()

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Get selected language
selected_lang = st.session_state.selected_language
lang_config = LANGUAGES[selected_lang]

# Display selection
st.info(f"🎯 **Currently translating:** English → {selected_lang}")

# Load model
with st.spinner(f"Loading {selected_lang} translation model..."):
    tokenizer, model, error = load_model(lang_config["model"])
    if error:
        st.error(f"Error loading model: {error}")
        st.warning("Some models may not be available. Please try another language.")
        st.stop()
    else:
        st.success(f"✅ {selected_lang} model loaded successfully!")

st.write("")

# Translation Interface
st.markdown("### 📝 Translation Interface")
col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown("**English Input**")
    input_text = st.text_area(
        "Enter your text",
        height=250,
        placeholder="Type your English text here...",
        key="input_text_area",
        label_visibility="collapsed"
    )

with col2:
    st.markdown(f"**{selected_lang} Translation**")
    output_placeholder = st.empty()

# Translate button
st.write("")
col_btn = st.columns([2, 1, 2])
with col_btn[1]:
    translate_btn = st.button("🚀 Translate", type="primary", use_container_width=True)

if translate_btn:
    if input_text.strip():
        with st.spinner("Translating..."):
            try:
                translation = translate_text(input_text, tokenizer, model)
                with col2:
                    output_placeholder.text_area(
                        "Translation result",
                        value=translation,
                        height=250,
                        key="output_text_area",
                        label_visibility="collapsed"
                    )
                st.success("✅ Translation completed successfully!")
            except Exception as e:
                st.error(f"Translation error: {str(e)}")
    else:
        st.warning("⚠️ Please enter some English text to translate")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Examples
st.markdown("### 💡 Quick Examples")
st.markdown("Click on any example to translate it instantly:")
st.write("")

cols = st.columns(5)
for idx, example in enumerate(EXAMPLE_SENTENCES):
    with cols[idx % 5]:
        if st.button(example, key=f"example_{idx}", use_container_width=True):
            with st.spinner("Translating..."):
                try:
                    translation = translate_text(example, tokenizer, model)
                    st.success("✅ Done!")
                    st.info(f"**English:** {example}\n\n**{selected_lang}:** {translation}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Footer
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("""
    <div class='footer'>
        <div class='developer-name'>
            💻 Developed by Hasnain Yaqoob
        </div>
        <div class='social-links'>
            <a href='https://www.linkedin.com/in/hasnainyaqoob' target='_blank'>LinkedIn</a> |
            <a href='https://x.com/Hasnain_Yaqoob_' target='_blank'>X (Twitter)</a> |
            <a href='https://github.com/hasnainyaqub' target='_blank'>GitHub</a> |
            <a href='https://www.kaggle.com/hasnainyaqooob' target='_blank'>Kaggle</a>
        </div>
        <div class='footer-text'>
            Powered by Helsinki-NLP OPUS-MT Models | Built with Streamlit & Transformers
        </div>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📦 Installation")
    st.code("pip install -r requirements.txt", language="bash")
    
    st.markdown("## 🚀 Quick Start")
    st.code("streamlit run app.py", language="bash")
    
    st.markdown("## 🌍 Features")
    st.markdown(f"""
    - ✅ **{len(LANGUAGES)} Languages** Supported
    - ✅ AI-Powered Translation
    - ✅ Real-time Processing
    - ✅ Modern & Clean UI
    - ✅ Fast & Accurate Results
    """)
    
    st.markdown("## ℹ️ About")
    st.info("""
        This translator uses state-of-the-art 
        Helsinki-NLP OPUS-MT models powered by 
        the Marian NMT framework.
        
        **Note:** Models are downloaded and 
        cached automatically on first use.
    """)
    
    st.markdown("## 📧 Contact")
    st.markdown("""
        **Hasnain Yaqoob**  
        AI/ML Developer
        
        Connect on social media for updates!
    """)