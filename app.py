import streamlit as st
from PIL import Image
from PIL.ExifTags import TAGS
import torch
import warnings
import urllib.parse
from transformers import AutoImageProcessor, AutoModelForImageClassification

# PAGE CONFIG
st.set_page_config(page_title="CyberFury | AI Forensic Lab", layout="wide", page_icon="🛡️")
warnings.filterwarnings("ignore", category=UserWarning)

# UI STYLING - UPDATED TO DEEP NAVY THEME
st.markdown(f"""
    <style>
    /* Main Background */
    .stApp {{ 
        background-color: #0F1C2E; 
        color: #E6EDF5; 
        font-family: 'Inter', -apple-system, sans-serif; 
    }}
    
    /* Headers */
    h1, h2, h3 {{ color: #E6EDF5 !important; font-weight: 700; }}
    
    /* File Uploader Customization */
    [data-testid="stFileUploader"] {{
        background-color: #16263D;
        padding: 20px;
        border-radius: 12px;
        border: 1px dashed #2F80ED;
    }}
    
    /* Browse Files Button Text Color */
    [data-testid="stFileUploader"] section button {{
        color: #FFFFFF !important;
        background-color: #2F80ED !important;
        border: none !important;
    }}
    
    /* Force upload instructional text to White/Light Gray */
    [data-testid="stFileUploader"] label, 
    [data-testid="stFileUploader"] p, 
    [data-testid="stFileUploader"] small {{
        color: #E6EDF5 !important;
    }}

    /* Action Buttons */
    .stButton>button {{ 
        width: 100%; 
        background-color: #2F80ED; 
        color: white; 
        border: none;
        border-radius: 8px;
        font-weight: bold;
        padding: 12px;
    }}
    .stButton>button:hover {{ 
        background-color: #56CCF2; 
        color: #0F1C2E;
    }}
    
    /* Report Cards */
    .status-box {{ 
        border: 1px solid #2F80ED; 
        padding: 24px; 
        background: #16263D; 
        border-radius: 12px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }}
    
    /* Probability Meter */
    .meter-container {{
        background-color: #0F1C2E; 
        height: 18px; 
        width: 100%;
        border-radius: 9px; 
        margin: 15px 0; 
        overflow: hidden;
    }}
    .meter-fill {{
        height: 100%; 
        transition: width 0.8s ease-in-out;
    }}
    
    /* Text Styles */
    .muted-text {{ color: #A0AEC0; font-size: 0.9em; }}
    
    .metadata-box {{
        border-left: 3px solid #56CCF2; 
        padding: 15px; 
        margin-top: 15px;
        background: rgba(86, 204, 242, 0.05); 
        border-radius: 0 8px 8px 0;
    }}
    </style>
""", unsafe_allow_html=True)

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
    def analyze(image, processor, model, device):
        if image.mode != 'RGB': image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        ai_score = probs[0].item()
        return {"verdict": "DEEPFAKE" if ai_score > 0.998 else "AUTHENTIC", "conf": ai_score * 100}

    @staticmethod
    def extract_metadata(image):
        metadata = {}
        suspicious_flags = []
        try:
            exif_data = image._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    metadata[tag_name] = str(value)
                    if tag_name == "Software":
                        ai_indicators = ["midjourney", "dalle", "stable diffusion", "automatic1111", "comfyui"]
                        if any(indicator in str(value).lower() for indicator in ai_indicators):
                            suspicious_flags.append(f"AI Signature: {value}")
            else:
                suspicious_flags.append("NO EXIF DATA FOUND")
        except:
            suspicious_flags.append("METADATA HEADER CORRUPTED")
        return {"metadata": metadata, "flags": suspicious_flags}

def main():
    st.markdown("<h1 style='text-align:center; padding-bottom: 30px;'>🛡️ CYBERFURY <span style='color:#56CCF2; font-weight:300;'>| AI Forensic Lab</span></h1>", unsafe_allow_html=True)
    
    if 'data' not in st.session_state: st.session_state.data = None
    if 'metadata' not in st.session_state: st.session_state.metadata = None

    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown("### 📥 Evidence Upload")
        file = st.file_uploader("", type=["jpg", "png", "jpeg", "webp"])
        
        if file:
            img = Image.open(file)
            st.image(img, use_container_width=True, caption="Current Forensic Subject")
            if st.button("🚨 RUN SCAN"):
                with st.spinner("Processing neural artifacts..."):
                    proc, mod, dev = CyberFuryEngine.load_model()
                    st.session_state.data = CyberFuryEngine.analyze(img, proc, mod, dev)
                    st.session_state.metadata = CyberFuryEngine.extract_metadata(img)

    with col2:
        st.markdown("### ⚖️ Analysis Report")
        
        # FIXED: Only process if BOTH file exists and analysis has been run
        if file is not None and st.session_state.data is not None:
            res = st.session_state.data
            score = res["conf"]
            
            # Theme Color Logic
            status_color = "#EB5757" if score > 90 else "#27AE60"
            
            st.markdown(f"""
                <div class='status-box'>
                    <h2 style='color: {status_color} !important; margin-top:0;'>{res['verdict']}</h2>
                    <p class='muted-text' style='margin-bottom:2px;'>SYNTHETIC PROBABILITY:</p>
                    <div class='meter-container'>
                        <div class='meter-fill' style='width: {score}%; background: {status_color};'></div>
                    </div>
                    <p style='text-align:right; font-weight:bold; color:{status_color};'>{score:.2f}% Confidence Score</p>
                </div>
            """, unsafe_allow_html=True)

            if score > 90:
                st.markdown(f"<p style='color:#EB5757; background:rgba(235, 87, 87, 0.1); padding:10px; border-radius:5px;'>🚩 <b>High Risk:</b> Synthetic patterns detected in pixel distribution.</p>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### 🛠️ Forensic Tools")
            
            # FIXED: query is now generated safely within the 'if file' block
            query = urllib.parse.quote(f'"{file.name}"')
            
            c1, c2 = st.columns(2)
            with c1:
                st.link_button("📂 Filename Origin", f"https://www.google.com/search?q={query}")
            with c2:
                st.link_button("🔍 Google Lens", "https://lens.google.com/upload")

            if st.session_state.metadata:
                meta = st.session_state.metadata
                if meta["flags"]:
                    st.markdown(f"<div class='metadata-box'><b style='color:#56CCF2;'>🚩 ANOMALY LOG:</b>", unsafe_allow_html=True)
                    for flag in meta["flags"]: st.markdown(f"<span style='font-size:0.85em;'>• {flag}</span>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("System Ready. Please upload evidence and click 'Run Scan'.")

if __name__ == "__main__":
    main()
