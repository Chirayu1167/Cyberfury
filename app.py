import streamlit as st
from PIL import Image
from PIL.ExifTags import TAGS
import torch
import time
import os
import warnings
import plotly.graph_objects as go
from transformers import AutoImageProcessor, AutoModelForImageClassification

#PAGE CONFIG
st.set_page_config(page_title="CyberFury | AI Forensic Lab", layout="wide", page_icon="⚡")
warnings.filterwarnings("ignore", category=UserWarning)
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #00f3ff; font-family: 'Courier New', Courier, monospace; }
    .stButton>button { 
        width: 100%; background: transparent; color: #00f3ff; border: 2px solid #00f3ff;
        font-weight: bold; box-shadow: 0 0 10px #00f3ff;
    }
    .reasoning-box {
        border: 1px solid #00f3ff; padding: 15px; background: rgba(0, 243, 255, 0.05);
        margin-top: 10px; border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

#THE ENGINE
class CyberFuryEngine:
    @staticmethod
    @st.cache_resource
    def load_model():
        model_name = "Organika/sdxl-detector"
        processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return processor, model, device

    @staticmethod
    def analyze(image, processor, model, device, buffer=0.90):
        #Preprocessing
        if image.mode != 'RGB':
            image = image.convert('RGB')

        #Inference
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Use Softmax to get 0.0 to 1.0 range
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        
        #Robust Score Extraction
        id2label = model.config.id2label
        ai_score = 0.0
        real_score = 0.0

        for idx, prob in enumerate(probs):
            label = id2label[idx].upper()
            val = prob.item()
            # If the label contains any "fake" related words, it's the AI score
            if any(key in label for key in ["AI", "FAKE", "SYNTHETIC", "ARTIFICIAL"]):
                ai_score = val
            # If it's the other label, it's the Real score
            else:
                real_score = val

        #Final Logic Decision
        if ai_score >= buffer:
            verdict = "AI"
            status = "🚩 AI-GENERATED DETECTED"
            conf = ai_score * 100
            reason = f"High-confidence AI signatures ({ai_score*100:.1f}%) exceed the 90% safety buffer."
        else:
            verdict = "REAL"
            status = "✨ AUTHENTIC / ORGANIC"
            # If we are calling it REAL, we show the Real confidence
            conf = real_score * 100
            if ai_score > 0.5:
                reason = f"Alert: AI signals detected ({ai_score*100:.1f}%), but dismissed as noise per 90% buffer rule."
            else:
                reason = "Pixel patterns appear consistently organic with no generative noise."

        return {
            "verdict": verdict, "status": status, "conf": conf, 
            "reason": reason, "ai_raw": ai_score, "real_raw": real_score
        }

#UI
def draw_gauge(val, verdict):
    color = "#ff003c" if verdict == "AI" else "#00f3ff"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val,
        number={'suffix': "%", 'font': {'color': color}},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color}, 'bgcolor': "#111"}
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#00f3ff"}, height=350)
    return fig

def main():
    st.markdown("<h1 style='text-align:center;'>⚡ CYBERFURY: NEON SCANNER</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    if 'data' not in st.session_state: st.session_state.data = None

    with col1:
        file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        if file:
            img = Image.open(file)
            st.image(img, use_container_width=True)
            if st.button("🚨 RUN FORENSIC SCAN"):
                proc, mod, dev = CyberFuryEngine.load_model()
                st.session_state.data = CyberFuryEngine.analyze(img, proc, mod, dev)

    with col2:
        if st.session_state.data:
            res = st.session_state.data
            st.plotly_chart(draw_gauge(res["conf"], res["verdict"]), use_container_width=True)
            
            color = "#ff003c" if res["verdict"] == "AI" else "#00f3ff"
            st.markdown(f"""
                <div style='border: 2px solid {color}; padding: 20px; box-shadow: 0 0 15px {color};'>
                    <h2 style='color: {color};'>{res['status']}</h2>
                    <p>{res['reason']}</p>
                    <small style='color: gray;'>Raw AI: {res['ai_raw']:.4f} | Raw Real: {res['real_raw']:.4f}</small>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("System Idle. Upload evidence to begin.")

if __name__ == "__main__":
    main()
