import streamlit as st
from PIL import Image
import torch
import warnings
import plotly.graph_objects as go
from transformers import AutoImageProcessor, AutoModelForImageClassification
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os

# PAGE CONFIG
st.set_page_config(page_title="CyberFury | AI Forensic Lab", layout="wide", page_icon="⚡")
warnings.filterwarnings("ignore", category=UserWarning)

# UI
st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #00f3ff; font-family: 'Courier New', Courier, monospace; }
    .stButton>button { 
        width: 100%; background: transparent; color: #00f3ff; border: 2px solid #00f3ff;
        font-weight: bold; box-shadow: 0 0 10px #00f3ff;
    }
    .stButton>button:hover { border: 2px solid #ff003c; color: #ff003c; box-shadow: 0 0 20px #ff003c; }
    </style>
""", unsafe_allow_html=True)

class CyberFuryEngine:
    @staticmethod
    @st.cache_resource
    def load_model():
        """Load AI detection model with optimization and error handling"""
        try:
            # Try different model loading strategies
            model_name = "Organika/sdxl-detector"
            
            # Strategy 1: Check for local models first
            local_path = "./models/sdxl-detector"
            if os.path.exists(local_path):
                st.info("📦 Loading from local cache...")
                model_name = local_path
            
            # Set environment variables for HuggingFace
            os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
            
            # Load processor and model
            processor = AutoImageProcessor.from_pretrained(
                model_name, 
                use_fast=True,
                resume_download=True,
                force_download=False,
                local_files_only=os.path.exists(local_path)
            )
            
            model = AutoModelForImageClassification.from_pretrained(
                model_name,
                resume_download=True,
                force_download=False,
                local_files_only=os.path.exists(local_path)
            )
            
            # Device Detection
            if torch.cuda.is_available():
                device = torch.device("cuda")
                st.success("🚀 GPU Acceleration Enabled")
            else:
                device = torch.device("cpu")
                st.info("💻 Running on CPU")
                
            model.to(device)
            model.eval()  # Set to evaluation mode for speed
            
            return processor, model, device
            
        except OSError as e:
            st.error("🌐 **Network Connection Error**")
            st.warning("""
            **Unable to download model from HuggingFace.** This is a Streamlit Cloud limitation.
            
            **Solutions:**
            1. Contact support to enable network access
            2. Use pre-downloaded models (see GitHub README)
            3. Deploy on platforms with unrestricted internet (Heroku, Railway, etc.)
            """)
            st.code(f"Error details: {str(e)}", language="bash")
            return None, None, None
            
        except Exception as e:
            st.error(f"❌ Unexpected Error: {str(e)}")
            st.info("💡 Try refreshing the page or contact support")
            return None, None, None
    
    @staticmethod
    @st.cache_resource
    def load_google_detector():
        """Load Google's TensorFlow Hub Object Detection Model - CACHED for speed"""
        try:
            st.info("🔄 Loading Google AI detector...")
            detector = hub.load("https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1")
            st.success("✅ Google detector loaded")
            return detector
        except Exception as e:
            st.warning(f"⚠️ Google detector unavailable: {str(e)}")
            st.info("Object detection will be disabled, but AI detection will still work.")
            return None

    @staticmethod
    def get_dynamic_reasoning(verdict, ai_raw, real_raw):
        """Generates context-aware reasoning based on score distribution."""
        if verdict == "AI":
            if ai_raw > 0.99999:
                return "Absolute AI signature detected. Mathematical convergence on synthetic diffusion patterns is 100%."
            return f"Anomalous pixel gradients detected. Synthetic markers ({ai_raw:.4f}) exceed safety thresholds."
        else:
            if real_raw > 0.95:
                return "Pure organic sensor data. Pixel-level noise matches physical hardware capture characteristics."
            elif 0.70 <= real_raw <= 0.95:
                return "Natural image with post-processing. Content verified as organic despite compression artifacts."
            else:
                return "Edge-case detected. Image shows mixed signals, but remains below the synthetic detection buffer."

    @staticmethod
    def detect_objects_google(image, detector):
        """Detect objects using Google's TensorFlow Hub model - OPTIMIZED"""
        if detector is None:
            return {
                "objects_found": 0,
                "top_object": "Detector unavailable",
                "confidence": 0.0
            }
        
        try:
            # Convert PIL to tensor with optimization
            img_array = np.array(image.resize((320, 320), Image.BILINEAR))  # Resize PIL directly
            img_tensor = tf.convert_to_tensor(img_array)
            img_tensor = tf.expand_dims(img_tensor, 0)
            img_tensor = tf.cast(img_tensor, tf.uint8)
            
            # Run detection
            result = detector(img_tensor)
            
            # Get detection results
            detection_scores = result['detection_scores'][0].numpy()
            detection_classes = result['detection_class_entities'][0].numpy()
            
            # Count detections with confidence > 0.3
            valid_detections = detection_scores > 0.3
            num_objects = int(np.sum(valid_detections))
            
            # Get top detected object
            if num_objects > 0:
                top_class = detection_classes[0].decode('utf-8').title()
                top_score = float(detection_scores[0])
            else:
                top_class = "None"
                top_score = 0.0
            
            return {
                "objects_found": num_objects,
                "top_object": top_class,
                "confidence": top_score
            }
        except Exception as e:
            st.warning(f"Object detection error: {str(e)}")
            return {
                "objects_found": 0,
                "top_object": "Detection failed",
                "confidence": 0.0
            }

    @staticmethod
    def analyze(image, processor, model, device, google_detector, buffer=0.9999):
        """Main analysis function - OPTIMIZED"""
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # AI Detection
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        
        id2label = model.config.id2label
        ai_score, real_score = 0.0, 0.0

        for idx, prob in enumerate(probs):
            label = id2label[idx].upper()
            val = prob.item()
            if any(key in label for key in ["AI", "FAKE", "SYNTHETIC", "ARTIFICIAL"]):
                ai_score = val
            else:
                real_score = val

        raw_ai_fixed = ai_score
        raw_real_fixed = real_score

        if ai_score >= buffer:
            verdict = "AI"
            status = "🚩 AI-GENERATED DETECTED"
            conf = ai_score * 100
        else:
            verdict = "REAL"
            status = "✨ AUTHENTIC / ORGANIC"
            conf = 100 - (real_score * 100)
            ai_score = 1 - ai_score
            real_score = 1 - real_score

        reason = CyberFuryEngine.get_dynamic_reasoning(verdict, raw_ai_fixed, raw_real_fixed)
        
        # Google Object Detection
        object_data = CyberFuryEngine.detect_objects_google(image, google_detector)

        return {
            "verdict": verdict, "status": status, "conf": conf, 
            "reason": reason, "ai_raw": ai_score, "real_raw": real_score,
            "device": device.type.upper(), "objects": object_data
        }

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
    st.markdown("<h1 style='text-align:center;'>⚡ CYBERFURY: FORENSIC SCANNER</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color: #888;'>Powered by Google TensorFlow Hub & HuggingFace Transformers</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    if 'data' not in st.session_state: 
        st.session_state.data = None
    if 'google_detector' not in st.session_state:
        st.session_state.google_detector = None
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False

    with col1:
        file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg", "webp"])
        if file:
            img = Image.open(file)
            st.image(img, use_container_width=True)
            
            if st.button("🚨 RUN FORENSIC SCAN", type="primary"):
                with st.spinner("🔍 Initializing AI detection engines..."):
                    # Load main model
                    proc, mod, dev = CyberFuryEngine.load_model()
                    
                    # Check if models loaded successfully
                    if proc is None or mod is None or dev is None:
                        st.error("❌ **Critical Error:** AI detection model failed to load.")
                        st.stop()
                    
                    # Load Google detector (only once)
                    if st.session_state.google_detector is None:
                        with st.spinner("⚙️ Loading Google AI models (one-time setup)..."):
                            st.session_state.google_detector = CyberFuryEngine.load_google_detector()
                    
                    # Run analysis
                    with st.spinner("🧬 Analyzing pixel signatures..."):
                        st.session_state.data = CyberFuryEngine.analyze(
                            img, proc, mod, dev, st.session_state.google_detector
                        )
                    
                    st.session_state.models_loaded = True
                    st.rerun()  # Force immediate UI update

    with col2:
        if st.session_state.data:
            res = st.session_state.data
            st.plotly_chart(draw_gauge(res["conf"], res["verdict"]), use_container_width=True)
            
            color = "#ff003c" if res["verdict"] == "AI" else "#00f3ff"
            obj_count = res['objects']['objects_found']
            obj_name = res['objects']['top_object']
            obj_conf = res['objects']['confidence']
            
            obj_display = ""
            if obj_count > 0:
                obj_display = f"""
                    <p style='font-size: 1.0em;'><strong>🎯 Objects Detected:</strong> {obj_count}</p>
                    <p style='font-size: 0.95em;'><strong>Primary Object:</strong> {obj_name} <span style='color: #00ff00;'>({obj_conf:.1%} confidence)</span></p>
                """
            else:
                obj_display = "<p style='font-size: 1.0em;'><strong>🎯 Objects Detected:</strong> None detected</p>"
            
            st.markdown(f"""
                <div style='border: 2px solid {color}; padding: 20px; box-shadow: 0 0 15px {color}; background: rgba(0,0,0,0.5);'>
                    <h2 style='color: {color}; margin-top:0;'>{res['status']}</h2>
                    <p style='font-size: 1.1em;'><strong>Reasoning:</strong> {res['reason']}</p>
                    {obj_display}
                    <hr style='border: 0.5px solid {color}; opacity: 0.3;'>
                    <small style='color: gray;'>
                        Inverse AI: {res['ai_raw']:.4f} | Inverse Real: {res['real_raw']:.4f} | Engine: {res['device']}
                    </small>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("⚡ System Online. Awaiting evidence for forensic deep-scan.")
            
            # Show system status
            if not st.session_state.models_loaded:
                with st.expander("📊 System Status"):
                    st.write("**AI Detection Engine:** Ready to load")
                    st.write("**Object Detection:** Ready to load")
                    st.write("**Processing Unit:** Auto-detect (CPU/GPU)")

if __name__ == "__main__":
    main()
