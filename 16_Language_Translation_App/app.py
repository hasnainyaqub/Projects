import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import torch

# Set page config
st.set_page_config(
    page_title="Multi-Language Translator",
    page_icon="🌐",
    layout="centered"
)

# Language configurations
LANGUAGES = {
    "English → Urdu": {
        "model": "Helsinki-NLP/opus-mt-en-ur",
        "src_lang": "English",
        "tgt_lang": "Urdu",
        "placeholder": "Type your English text here...",
        "examples": [
            "Hello, how are you?",
            "I am fine, thank you",
            "The weather is very nice today",
            "What is your name?",
            "Good morning"
        ]
    },
    "English → French": {
        "model": "Helsinki-NLP/opus-mt-en-fr",
        "src_lang": "English",
        "tgt_lang": "French",
        "placeholder": "Type your English text here...",
        "examples": [
            "Hello, how are you?",
            "Thank you very much",
            "Where is the train station?",
            "I love learning languages",
            "Good evening"
        ]
    },
    "English → Spanish": {
        "model": "Helsinki-NLP/opus-mt-en-es",
        "src_lang": "English",
        "tgt_lang": "Spanish",
        "placeholder": "Type your English text here...",
        "examples": [
            "Hello, how are you?",
            "Thank you very much",
            "I am learning Spanish",
            "What time is it?",
            "Have a nice day"
        ]
    },
    "English → German": {
        "model": "Helsinki-NLP/opus-mt-en-de",
        "src_lang": "English",
        "tgt_lang": "German",
        "placeholder": "Type your English text here...",
        "examples": [
            "Hello, how are you?",
            "Thank you very much",
            "I am learning German",
            "Where is the bathroom?",
            "Good night"
        ]
    }
}

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_model(model_name):
    """Load the translation model and tokenizer"""
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate_text(text, tokenizer, model):
    """Translate text using the loaded model"""
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Generate translation
    with torch.no_grad():
        translated = model.generate(**inputs)
    
    # Decode the translation
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# App UI
st.title("🌐 Multi-Language Translator")
st.markdown("Translate English to multiple languages using AI-powered neural machine translation")

# Language selection
selected_language = st.selectbox(
    "Select Translation Direction:",
    list(LANGUAGES.keys()),
    index=0
)

lang_config = LANGUAGES[selected_language]

# Load model
with st.spinner(f"Loading {selected_language} translation model..."):
    try:
        tokenizer, model = load_model(lang_config["model"])
        st.success(f"✓ {selected_language} model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"{lang_config['src_lang']} Input")
    input_text = st.text_area(
        f"Enter {lang_config['src_lang']} text:",
        height=200,
        placeholder=lang_config['placeholder'],
        key="input_text"
    )

with col2:
    st.subheader(f"{lang_config['tgt_lang']} Translation")
    translation_placeholder = st.empty()

# Translate button
if st.button("Translate 🔄", type="primary", use_container_width=True):
    if input_text.strip():
        with st.spinner("Translating..."):
            try:
                translation = translate_text(input_text, tokenizer, model)
                with col2:
                    translation_placeholder.text_area(
                        "Translation:",
                        value=translation,
                        height=200,
                        key="output_text"
                    )
            except Exception as e:
                st.error(f"Translation error: {str(e)}")
    else:
        st.warning(f"Please enter some {lang_config['src_lang']} text to translate")

# Example texts
st.markdown("---")
st.subheader("Example Translations")
st.markdown("Click on an example to try it:")

examples = lang_config['examples']
cols = st.columns(3)
for idx, example in enumerate(examples):
    with cols[idx % 3]:
        if st.button(example, key=f"example_{idx}", use_container_width=True):
            with st.spinner("Translating..."):
                try:
                    translation = translate_text(example, tokenizer, model)
                    st.success("Translation complete!")
                    st.info(f"**{lang_config['src_lang']}:** {example}\n\n**{lang_config['tgt_lang']}:** {translation}")
                except Exception as e:
                    st.error(f"Translation error: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Powered by Helsinki-NLP OPUS-MT Models | Built with Streamlit & Transformers</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Installation instructions in sidebar
with st.sidebar:
    st.header("📦 Installation")
    st.code("""
pip install streamlit
pip install transformers
pip install torch
pip install sentencepiece
    """, language="bash")
    
    st.header("🚀 How to Run")
    st.code("""
streamlit run app.py
    """, language="bash")
    
    st.header("🌍 Supported Languages")
    st.markdown("""
    - English → Urdu
    - English → French
    - English → Spanish
    - English → German
    """)
    
    st.header("ℹ️ About")
    st.info(
        """
        This app uses Helsinki-NLP OPUS-MT models 
        for translation. The models are based on 
        the Marian NMT framework and trained on 
        parallel corpora from OPUS.
        """
    )