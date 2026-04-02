import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import joblib
import matplotlib.pyplot as plt

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Mars Colony AI", layout="centered")

# ------------------ DARK THEME ------------------
st.markdown("""
<style>
.stApp {
    background-color: #0b0f1a;
    color: white;
}
.glow {
    color: #00e5ff;
    text-shadow: 0 0 10px #00e5ff;
}
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODELS (CACHED) ------------------
@st.cache_resource
def load_models():
    try:
        model_ml = joblib.load("model_ml.pkl")
        scaler = joblib.load("scaler.pkl")

        model_dl = models.resnet18(weights=None)
        model_dl.fc = nn.Linear(model_dl.fc.in_features, 2)

        model_dl.load_state_dict(
            torch.load("resnet_model.pth", map_location=torch.device("cpu"))
        )

        model_dl.eval()

        return model_ml, scaler, model_dl

    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None, None


model_ml, scaler, model_dl = load_models()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ------------------ FUNCTIONS ------------------
def predict_image(image):
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model_dl(img)
        _, pred = torch.max(output, 1)

    return pred.item()


def final_prediction(terrain, infra, image):
    input_df = pd.DataFrame(
        [[terrain, infra]],
        columns=['terrain_risk', 'infra_score']
    )

    tab_input = scaler.transform(input_df)
    tab_score = model_ml.predict(tab_input)[0]

    img_score = predict_image(image)

    final_score = 0.5 * tab_score + 0.3 * infra + 0.2 * img_score

    return final_score, tab_score, img_score


# ------------------ HEADER ------------------
st.markdown("<h1 class='glow'>🚀 MARS COLONY AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Multimodal Decision Intelligence System</p>", unsafe_allow_html=True)

st.markdown("---")

# ------------------ INPUT ------------------
col1, col2 = st.columns(2)

with col1:
    terrain = st.slider("Terrain Risk", 0.0, 1.0, 0.5)

with col2:
    infra = st.slider("Infrastructure Score", 0.0, 1.0, 0.5)

st.markdown("### 🛰 Upload Terrain Image")
uploaded_file = st.file_uploader("", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)
else:
    image = None

st.markdown("---")

# ------------------ ACTION ------------------
if st.button("🚀 RUN ANALYSIS"):

    if model_ml is None:
        st.error("Models not loaded properly.")
    elif image is None:
        st.error("Upload an image first")
    else:
        final_score, tab_score, img_score = final_prediction(terrain, infra, image)

        # ------------------ RESULT ------------------
        st.markdown("## 🧠 AI DECISION")

        if final_score > 0.7:
            status = "HIGHLY SUITABLE"
            color = "#00ffcc"
        elif final_score > 0.4:
            status = "MODERATE"
            color = "#ffaa00"
        else:
            status = "NOT SUITABLE"
            color = "#ff4d4d"

        st.markdown(f"<h2 style='color:{color}; text-align:center;'>{status}</h2>", unsafe_allow_html=True)

        st.progress(int(final_score * 100))
        st.markdown(f"<h3 style='text-align:center;'>Score: {final_score:.2f}</h3>", unsafe_allow_html=True)

        st.markdown("---")

        # ------------------ LIVE CHART ------------------
        st.markdown("### 📊 Score Breakdown")

        labels = ['Tabular Model', 'Infrastructure', 'Image Model']
        values = [tab_score, infra, img_score]

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        ax.set_title("Model Contribution")

        st.pyplot(fig)

        # ------------------ EXPLAINABILITY ------------------
        st.markdown("### 🧠 Why this decision?")

        if terrain > 0.7:
            st.warning("Terrain is highly unstable")
        else:
            st.success("Terrain conditions are stable")

        if infra < 0.3:
            st.warning("Infrastructure support is weak")
        else:
            st.success("Infrastructure is sufficient")

        if img_score == 1:
            st.warning("Visual model detected hazardous terrain")
        else:
            st.success("Visual terrain appears safe")

        st.markdown("---")

        # ------------------ CONFIDENCE ------------------
        st.markdown("### 🔬 Model Confidence")

        confidence = abs(tab_score - img_score)
        st.progress(int((1 - confidence) * 100))

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>AI + Deep Learning + Space Intelligence</p>", unsafe_allow_html=True)
