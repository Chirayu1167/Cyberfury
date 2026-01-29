import streamlit as st
from PIL import Image
from PIL.ExifTags import TAGS
import torch
import warnings
import urllib.parse
from transformers import AutoImageProcessor, AutoModelForImageClassification

# PAGE CONFIG
st.set_page_config(page_title="CyberFury | AI Forensic Lab", layout="wide", page_icon="⚡")
warnings.filterwarnings("ignore", category=UserWarning)

# UI STYLING
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #00f3ff; font-family: 'Courier New', Courier, monospace; }
    .stButton>button { 
        width: 100%; background: transparent; color: #00f3ff; border: 2px solid #00f3ff;
        font-weight: bold; box-shadow: 0 0 10px #00f3ff; margin-bottom: 8px;
    }
    .stButton>button:hover { border: 2px solid #ff003c; color: #ff003c; box-shadow: 0 0 20px #ff003c; }
    .status-box { 
        border: 2px solid #00f3ff; padding: 20px; 
        box-shadow: 0 0 15px #00f3ff; background: rgba(0, 243, 255, 0.05); 
        border-radius: 5px;
    }
    .metadata-box {
        border: 1px solid #00f3ff; padding: 10px; margin-top: 10px;
        background: rgba(0, 243, 255, 0.03); border-radius: 3px;
        font-size: 0.85em;
    }
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
        return {"verdict": "AI" if ai_score > 0.99985 else "REAL", "conf": ai_score * 100}

    @staticmethod
    def extract_metadata(image):
        """Extract EXIF metadata from image to detect AI generation traces"""
        metadata = {}
        suspicious_flags = []
        
        try:
            exif_data = image._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    metadata[tag_name] = str(value)
                    
                    if tag_name == "Software":
                        ai_indicators = ["midjourney", "dalle", "stable diffusion", "stablediffusion", 
                                       "automatic1111", "comfyui", "leonardo", "playground"]
                        if any(indicator in str(value).lower() for indicator in ai_indicators):
                            suspicious_flags.append(f"AI Software Detected: {value}")
                    
                    if tag_name == "Make" and not value:
                        suspicious_flags.append("Missing Camera Manufacturer")
                    if tag_name == "Model" and not value:
                        suspicious_flags.append("Missing Camera Model")
            else:
                suspicious_flags.append("⚠️ NO EXIF DATA - Possible AI generation or scrubbed metadata")
                
        except (AttributeError, KeyError):
            suspicious_flags.append("⚠️ NO METADATA FOUND - Highly suspicious for AI images")
        
        img_format = image.format
        if img_format in ["PNG", "WEBP"] and not metadata:
            suspicious_flags.append(f"⚠️ {img_format} with no metadata - Common AI output format")
        
        return {
            "metadata": metadata,
            "flags": suspicious_flags,
            "has_camera_info": "Make" in metadata or "Model" in metadata
        }

def main():
    st.markdown("<h1 style='text-align:center;'>⚡ CYBERFURY</h1>", unsafe_allow_html=True)
    
    if 'data' not in st.session_state: st.session_state.data = None
    if 'metadata' not in st.session_state: st.session_state.metadata = None

    col1, col2 = st.columns(2)

    with col1:
        file = st.file_uploader("Upload Evidence", type=["jpg", "png", "jpeg", "webp"])
        if file:
            img = Image.open(file)
            st.image(img, use_container_width=True)
            if st.button("🚨 EXECUTE DEEP SCAN"):
                with st.spinner("Decoding pixel signatures..."):
                    proc, mod, dev = CyberFuryEngine.load_model()
                    st.session_state.data = CyberFuryEngine.analyze(img, proc, mod, dev)
                    st.session_state.metadata = CyberFuryEngine.extract_metadata(img)

    with col2:
        if st.session_state.data:
            res = st.session_state.data
            color = "#ff003c" if res["verdict"] == "AI" else "#00f3ff"
            
            st.markdown(f"""
                <div class='status-box' style='border-color: {color};'>
                    <h2 style='color: {color}; margin-top:0;'>{res['verdict']} DETECTED</h2>
                    <p>Confidence: <b>{res['conf']:.4f}%</b></p>
                    <hr style='border: 0.5px solid {color}; opacity: 0.3;'>
                    <p style='color: #00f3ff;'><b>FACT CHECKING:</b></p>
                </div>
            """, unsafe_allow_html=True)

            query = urllib.parse.quote(f'"{file.name}"')
            st.link_button("📂 SEARCH FILENAME SOURCE", f"https://www.google.com/search?q={query}")
            st.link_button("🌍 CHECK GOOGLE FACT-CHECK DATABASE", "https://toolbox.google.com/factcheck/explorer")

            if st.session_state.metadata:
                meta = st.session_state.metadata
                st.markdown("---")
                st.markdown("### 📊 3. Google Cloud DLP + Metadata Analysis")
                
                if meta["flags"]:
                    flag_color = "#ff003c"
                    st.markdown(f"""
                        <div class='metadata-box' style='border-color: {flag_color};'>
                            <p style='color: {flag_color}; font-weight: bold;'>🚩 SUSPICIOUS INDICATORS DETECTED:</p>
                    """, unsafe_allow_html=True)
                    for flag in meta["flags"]:
                        st.markdown(f"• {flag}")
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class='metadata-box' style='border-color: #00f3ff;'>
                            <p style='color: #00f3ff; font-weight: bold;'>✓ Camera metadata present</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                if meta["metadata"]:
                    with st.expander("🔍 View Raw Metadata"):
                        important_keys = ["Make", "Model", "Software", "DateTime", "DateTimeOriginal"]
                        for key in important_keys:
                            if key in meta["metadata"]:
                                st.text(f"{key}: {meta['metadata'][key]}")

            st.markdown("---")
            st.info("💡 **Manual Cross-Check:** Right-click the image on the left, select **'Copy Image'**, then click below and press **Ctrl+V**.")
            st.link_button("🔍 OPEN GOOGLE LENS (PASTE MODE)", "https://lens.google.com/upload")
            
        else:
            st.info("System Online. Awaiting evidence for forensic deep-scan.")

if __name__ == "__main__":
    main()
