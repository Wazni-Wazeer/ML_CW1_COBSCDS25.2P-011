"""
NHPT Heritage AI — Interactive Streamlit Application
Built with Streamlit, PyTorch (EfficientNet-B0), Chroma, and Ollama (Llama 3.2).

Usage:
    streamlit run streamlit_app.py
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st

# LangChain imports with fallback
try:
    from langchain_ollama import ChatOllama, OllamaEmbeddings
except (ImportError, AttributeError, ModuleNotFoundError):
    from langchain_community.chat_models import ChatOllama
    from langchain_community.embeddings import OllamaEmbeddings

try:
    from langchain_chroma import Chroma
except (ImportError, ModuleNotFoundError):
    from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ModuleNotFoundError:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuration & Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
MODEL_PATH = BASE_DIR / "models" / "best_model.pth"
META_PATH = BASE_DIR / "models" / "model_metadata.json"
CHROMA_DIR = BASE_DIR / "chroma_db"

LLM_MODEL = "llama3.2"
LLM_TEMP = 0.3
EMBEDDING_MODEL = "nomic-embed-text"
RELEVANCE_THRESHOLD = 0.3
LOW_CONF_THRESHOLD = 0.70

OOD_CONFIDENCE_THRESHOLD = 0.45
OOD_ENTROPY_RATIO = 0.85

# Page Configuration
st.set_page_config(
    page_title="NHPT Heritage AI Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .card-container {
        background-color: #0F172A;
        border-radius: 12px;
        padding: 20px;
        color: #FFFFFF;
        border: 1px solid #1E293B;
        margin-bottom: 15px;
    }
    .badge-green {
        background-color: #16A34A;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-orange {
        background-color: #D97706;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-red {
        background-color: #DC2626;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Resource Caching (CV Model & RAG Pipeline)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_cv_model():
    """Load and cache PyTorch CV Model."""
    if not META_PATH.exists() or not MODEL_PATH.exists():
        st.error("Model metadata or weights not found in ./models/")
        st.stop()

    with open(META_PATH, "r", encoding="utf-8") as f:
        model_meta = json.load(f)

    arch_name = model_meta.get("model_architecture", "efficientnet_b0").lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = model_meta["class_names"]
    display_names = model_meta.get("display_names", class_names)
    num_classes = model_meta["num_classes"]

    if arch_name == "convnext_tiny":
        cv_model = models.convnext_tiny(weights=None)
        in_features = cv_model.classifier[2].in_features
        cv_model.classifier[2] = nn.Sequential(
            nn.Dropout(p=model_meta.get("dropout", 0.3)),
            nn.Linear(in_features, num_classes)
        )
    elif arch_name == "efficientnet_b2":
        cv_model = models.efficientnet_b2(weights=None)
        in_features = cv_model.classifier[1].in_features
        cv_model.classifier = nn.Sequential(
            nn.Dropout(p=model_meta.get("dropout", 0.2), inplace=True),
            nn.Linear(in_features, num_classes)
        )
    else:
        cv_model = models.efficientnet_b0(weights=None)
        in_features = cv_model.classifier[1].in_features
        cv_model.classifier = nn.Sequential(
            nn.Dropout(p=model_meta.get("dropout", 0.3), inplace=True),
            nn.Linear(in_features, num_classes)
        )

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    cv_model.load_state_dict(checkpoint["model_state_dict"])
    cv_model = cv_model.to(device)
    cv_model.eval()

    return cv_model, model_meta, device, class_names, display_names


@st.cache_resource
def load_rag_pipeline():
    """Load and cache Chroma Vector Store & LangChain RunnableWithMessageHistory chain."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url="http://localhost:11434")
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="nhpt_heritage"
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    SYSTEM_PROMPT = """You are a knowledgeable heritage guide for the National Heritage Preservation Trust (NHPT). 
Your role is to provide clear, well-structured explanations of architectural styles and historic sites for visitors.

RESPONSE STRUCTURE REQUIREMENTS:
1. Start with an introductory overview explanation of the architectural style, its historical period/origins, and overall significance.
2. Follow with detailed key features, structural characteristics, materials, and construction techniques (formatted clearly with key sections or bullet points).
3. If image classification details (predicted style name and confidence score) are present in the query:
   - State the prediction and confidence percentage clearly at the beginning.
   - If confidence is below 70%, explicitly note the uncertainty and mention alternative candidate styles.
4. ALWAYS cite the source document by name (e.g., "According to the source on Ancient Egyptian Architecture...").
5. Strictly use ONLY information provided in the CONTEXT below. If the context does not contain enough information, state: "I don't have enough information in my knowledge base to answer that question. I can only provide information about the architectural styles in my database."

CONTEXT FROM KNOWLEDGE BASE:
{context}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMP, base_url="http://localhost:11434")
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    session_store = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in session_store:
            session_store[session_id] = InMemoryChatMessageHistory()
        return session_store[session_id]

    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return vectorstore, conversational_chain, session_store


# Load cached resources
cv_model, model_meta, device, class_names, display_names = load_cv_model()
vectorstore, conversational_chain, session_store = load_rag_pipeline()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Helper Logic: OOD Detection & Classification with TTA
# ─────────────────────────────────────────────────────────────────────────────

def _is_ood(probs: np.ndarray) -> bool:
    max_conf = float(probs.max())
    if max_conf < OOD_CONFIDENCE_THRESHOLD:
        return True
    eps = 1e-10
    entropy = -np.sum(probs * np.log(probs + eps))
    max_entropy = np.log(len(probs))
    entropy_ratio = entropy / max_entropy
    return entropy_ratio > OOD_ENTROPY_RATIO


def classify_image_tta(image: Image.Image) -> Tuple[Optional[dict], bool]:
    """Run Multi-Crop TTA + Letterboxing inference."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    img_size = model_meta.get("img_size", 224)
    norm = transforms.Normalize(
        mean=model_meta.get("imagenet_mean", [0.485, 0.456, 0.406]),
        std=model_meta.get("imagenet_std", [0.229, 0.224, 0.225])
    )

    w, h = image.size
    max_dim = max(w, h)
    letterbox = Image.new("RGB", (max_dim, max_dim), (128, 128, 128))
    letterbox.paste(image, ((max_dim - w) // 2, (max_dim - h) // 2))

    t_letterbox = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(), norm])
    t_letterbox_flip = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), norm])
    t_center = transforms.Compose([transforms.Resize(img_size + 32), transforms.CenterCrop(img_size), transforms.ToTensor(), norm])

    t_resized = transforms.Resize((img_size + 32, img_size + 32))(image)
    crop_left = t_resized.crop((0, 16, img_size, img_size + 16))
    crop_right = t_resized.crop((32, 16, img_size + 32, img_size + 16))
    t_crop = transforms.Compose([transforms.ToTensor(), norm])

    tensors = [
        t_letterbox(letterbox),
        t_letterbox_flip(letterbox),
        t_center(image),
        t_crop(crop_left),
        t_crop(crop_right)
    ]
    batch = torch.stack(tensors).to(device)

    with torch.no_grad():
        outputs = cv_model(batch)
        probs_all = torch.softmax(outputs / 0.65, dim=1).cpu().numpy()
        weights = np.array([0.35, 0.25, 0.20, 0.10, 0.10])
        probs = np.average(probs_all, axis=0, weights=weights)

    if _is_ood(probs):
        return None, True

    top_idx = int(probs.argmax())
    raw_style = class_names[top_idx]
    predicted_style = display_names.get(raw_style, raw_style) if isinstance(display_names, dict) else raw_style
    confidence = float(probs[top_idx])

    top3_indices = probs.argsort()[::-1][:3]
    top3 = []
    for i in top3_indices:
        raw_s = class_names[i]
        disp_s = display_names.get(raw_s, raw_s) if isinstance(display_names, dict) else raw_s
        top3.append((disp_s, float(probs[i])))

    return {
        "predicted_style": predicted_style,
        "confidence": confidence,
        "top_3": top3
    }, False

# ─────────────────────────────────────────────────────────────────────────────
# 4. Streamlit UI Layout
# ─────────────────────────────────────────────────────────────────────────────

# Header
st.markdown("<div class='main-header'>NHPT Heritage AI Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>National Heritage Preservation Trust — Architectural Style Classification & Knowledge Support</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("System Specifications")
    st.markdown("""
    - **Vision Backbone**: EfficientNet-B0 (5.3M params)
    - **Inference Mode**: Multi-Crop TTA + Letterboxing
    - **Vector DB**: Chroma (`nhpt_heritage`)
    - **Embeddings**: `nomic-embed-text` (768-dim)
    - **LLM**: Llama 3.2 (3B parameters via Ollama)
    - **Memory**: Per-session Chat History
    """)
    st.divider()
    
    st.subheader("Supported Architectural Styles")
    for style in [
        "21st Century Eco Architecture",
        "21st Century International Style",
        "Ancient Egyptian Architecture",
        "Herodian Architecture",
        "Colonial Revival Architecture",
        "Roman Classical Architecture"
    ]:
        st.markdown(f"- {style}")

    st.divider()
    if st.button("Clear Chat History", type="secondary", use_container_width=True):
        st.session_state.messages = []
        if "session_id" in st.session_state and st.session_state.session_id in session_store:
            session_store[st.session_state.session_id] = InMemoryChatMessageHistory()
        st.rerun()

# Session State Initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = f"streamlit_{int(time.time())}"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_pred" not in st.session_state:
    st.session_state.last_pred = None

# Main Two-Column Layout
col1, col2 = st.columns([4, 6], gap="large")

# ── COLUMN 1: Image Upload & Vision Model Classification ──────────────────────
with col1:
    st.subheader("1. Upload Building Photo")
    uploaded_file = st.file_uploader(
        "Choose a photograph of a building or artifact...",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Analyze Architectural Style", type="primary", use_container_width=True):
            with st.spinner("Evaluating multi-crop visual features..."):
                pred_dict, is_ood = classify_image_tta(image)
                
                if is_ood:
                    st.warning("### Not a Recognized Architectural Style\nThis photo does not match any of the 6 supported architectural styles in our knowledge base.")
                    st.session_state.last_pred = None
                else:
                    st.session_state.last_pred = pred_dict

    # Render Prediction Card if prediction exists
    if st.session_state.last_pred:
        pred = st.session_state.last_pred
        style = pred["predicted_style"]
        conf = pred["confidence"]
        top3 = pred["top_3"]

        badge_class = "badge-green" if conf >= 0.8 else ("badge-orange" if conf >= 0.7 else "badge-red")
        
        st.markdown(f"""
        <div class='card-container'>
            <h3 style='margin-top:0; color:#38BDF8;'>Architectural Style Prediction</h3>
            <p style='font-size: 1.25rem; font-weight: 800; margin-bottom: 12px;'>
                {style} <span class='{badge_class}'>{conf:.1%} Confidence</span>
            </p>
            <hr style='border: 0; border-top: 1px solid #334155; margin: 12px 0;'>
            <p style='margin-bottom: 8px; font-weight: 700; color: #94A3B8;'>Top Candidate Predictions:</p>
        </div>
        """, unsafe_allow_html=True)

        for s_name, score in top3:
            st.write(f"**{s_name}**: `{score:.1%}`")
            st.progress(score)

# ── COLUMN 2: LangChain RAG Conversational Assistant ──────────────────────────
with col2:
    st.subheader("2. Ask Heritage Assistant")

    # Render conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if user_input := st.chat_input("Ask a question about this building or heritage style..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Build prompt input with CV prediction enrichment if present
        query = user_input
        pred_dict = st.session_state.last_pred
        if pred_dict:
            style = pred_dict["predicted_style"]
            conf = pred_dict["confidence"]
            top3 = pred_dict["top_3"]
            top3_str = ", ".join([f"{s} ({p:.0%})" for s, p in top3])

            if conf < LOW_CONF_THRESHOLD:
                query = (
                    f"An image of a building was classified as '{style}' with LOW confidence ({conf:.0%}). "
                    f"The top-3 predictions were: {top3_str}. "
                    f"Please explicitly note the uncertainty to the visitor, and describe key visual features distinguishing '{style}'. "
                    f"Visitor's question: {user_input}"
                )
            else:
                query = (
                    f"An image of a building was classified as '{style}' with {conf:.0%} confidence. "
                    f"Visitor's question: {user_input}"
                )

        # Hallucination Guard Relevance Check
        docs = vectorstore.similarity_search(query, k=1)
        if not docs:
            no_info = (
                "I don't have enough information in my knowledge base to answer that question. "
                "I can only provide details about these 6 architectural styles: 21st Century Eco Architecture, "
                "International Style, Ancient Egyptian, Herodian, Colonial Revival, and Roman Classical Architecture."
            )
            st.session_state.messages.append({"role": "assistant", "content": no_info})
            with st.chat_message("assistant"):
                st.markdown(no_info)
        else:
            with st.chat_message("assistant"):
                with st.spinner("Searching NHPT knowledge base & synthesizing cited response..."):
                    res = conversational_chain.invoke(
                        {"input": query},
                        config={"configurable": {"session_id": st.session_state.session_id}}
                    )
                    answer = res["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
