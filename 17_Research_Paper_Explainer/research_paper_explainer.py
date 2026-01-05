import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import load_prompt
import json

# --------------------------------
# Page Config
# --------------------------------
st.set_page_config(
    page_title="AI Research Explainer",
    page_icon="📘",
    layout="wide"
)

# --------------------------------
# API Key Setup
# --------------------------------
api_key = st.secrets['GROQ_API_KEY']

headers = {
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json"
    }

# --------------------------------
# Model
# --------------------------------
model = ChatGroq(model="openai/gpt-oss-120b")

# --------------------------------
# Custom Light Mode CSS
# --------------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
        color: #0f172a;
    }

    h1, h2, h3 {
        color: #020617;
    }

    .card {
        background: #ffffff;
        border-radius: 16px;
        padding: 26px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 20px rgba(0,0,0,0.06);
        margin-bottom: 24px;
    }

    .result-card {
        background: #ffffff;
        border-left: 6px solid #2563eb;
        border-radius: 14px;
        padding: 26px;
        margin-top: 28px;
        line-height: 1.7;
        box-shadow: 0 8px 20px rgba(0,0,0,0.06);
    }

    .stButton > button {
        background: linear-gradient(90deg, #2563eb, #3b82f6);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
        border: none;
        transition: all 0.2s ease-in-out;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #1d4ed8, #2563eb);
        box-shadow: 0 6px 16px rgba(37,99,235,0.35);
        transform: translateY(-1px);
    }

    .stSelectbox label {
        font-weight: 600;
        color: #020617;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------
# Header
# --------------------------------
st.markdown(
    """
    <div class="card">
        <h1>📘 AI Research Explainer</h1>
        <p>
        Select a research paper and generate a clear, structured explanation
        using advanced language models.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------
# Layout
# --------------------------------
left_col, right_col = st.columns([2, 1])

with left_col:
    paper_input = st.selectbox(
        "📄 Research Paper",
        [
            "Attention Is All You Need",
            "BERT: Pre-training of Deep Bidirectional Transformers",
            "GPT-3: Language Models are Few-Shot Learners",
            "Diffusion Models Beat GANs on Image Synthesis",
            "Vision Transformer ViT",
            "CLIP: Connecting Text and Images",
            "PaLM: Scaling Language Modeling",
            "LoRA: Low Rank Adaptation of Large Language Models",
            "DINO: Self-Supervised Vision Learning",
            "UNet: Convolutional Networks for Biomedical Image Segmentation",
            "ResNet: Deep Residual Learning",
            "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context",
            "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
            "ALBERT: A Lite BERT for Self-supervised Learning",
            "T5: Exploring the Limits of Transfer Learning",
            "XLNet: Generalized Autoregressive Pretraining",
            "ELECTRA: Pre-training Text Encoders as Discriminators",
            "LLaMA: Open and Efficient Foundation Language Models",
            "Mamba State Space Models",
            "SAM: Segment Anything Model",
            "Stable Diffusion",
            "NeRF",
            "StyleGAN2",
            "YOLOv5 Research Overview",
            "Swin Transformer",
            "DETR",
            "EfficientNet",
            "Layer Normalization",
            "RMSNorm",
            "Graph Attention Networks (GAT)",
            "Neural ODEs",
            "GAN: Generative Adversarial Networks",
            "AlphaFold",
            "PPO: Proximal Policy Optimization",
            "SimCLR",
            "MoCo",
            "FlashAttention",
            "GPT-4 Technical Report",
            "DALL·E",
            "ControlNet",
            "Whisper",
            "Hybrid CNN Transformer Architectures",
        ]
    )

with right_col:
    style_input = st.selectbox(
        "🎨 Explanation Style",
        [
            "Beginner-Friendly",
            "Technical",
            "Code-Oriented",
            "Mathematical",
            "Visual Analogy Based",
            "Real World Examples",
            "Step by Step Breakdown",
            "Use Case Focused",
        ]
    )

    length_input = st.selectbox(
        "📏 Explanation Length",
        [
            "Short (1 to 2 paragraphs)",
            "Medium (3 to 5 paragraphs)",
            "Long (detailed explanation)",
            "Bullet Points Only",
            "Extended Deep Dive",
        ]
    )

# --------------------------------
# Load Prompt
# --------------------------------
prompt_template = load_prompt(
    "00_Projects/01_Research_Paper_Explainer/Efficient_prompt.json"
)

# --------------------------------
# Generate Button
# --------------------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("✨ Generate Explanation"):
    with st.spinner("Generating explanation..."):
        chain = prompt_template | model
        response = chain.invoke(
            {
                "paper_input": paper_input,
                "style_input": style_input,
                "length_input": length_input,
            }
        )

    st.markdown(
        f"""
        <div class="result-card">
            <h3>📖 Generated Explanation</h3>
            <p>{response.content}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
