# ⚡ CyberFury: AI Forensic Lab

**CyberFury** is a high-precision forensic suite designed to distinguish between authentic photography and AI-generated imagery. It uses a specialized Swin Transformer architecture to identify generative signatures while utilizing a **90% Confidence Buffer** to protect real images from being falsely flagged due to social media compression.

### 🔍 Forensic Logic & The 90% Buffer
Standard detectors often fail on "noisy" real photos (e.g., WhatsApp images). CyberFury fixes this by requiring extreme certainty before a "Fake" verdict:
- **90% - 100% Score:** 🚩 **AI-GENERATED** (High-confidence generative signatures detected).
- **Below 90% Score:** ✨ **AUTHENTIC** (Signals are dismissed as digital noise or compression artifacts).



### 🛠️ Tech Stack & Features
- **Neural Engine:** Swin Transformer (`sdxl-detector`) via HuggingFace & PyTorch.
- **Dynamic UI:** Neon-themed Streamlit interface with animated Plotly gauges.
- **Hardware Audit:** Scans EXIF metadata for legitimate camera signatures (Apple, Samsung, etc.).
- **Transparent Reasoning:** Provides a logic log explaining the verdict in plain English.

### 🚀 Quick Deployment
1. **GitHub:** Upload `app.py` and `requirements.txt` (include: `streamlit`, `torch`, `transformers`, `pillow`, `plotly`, `accelerate`).
2. **Streamlit Cloud:** Connect your repo, set Python to 3.9+, and Deploy.
---
**Developed by [Chirayu](https://github.com/chirayu1167)**
