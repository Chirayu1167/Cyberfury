# ⚡ CyberFury – AI Forensic Image Scanner

CyberFury is a **forensic-grade AI image analysis application** built with **Streamlit, PyTorch, and Hugging Face Transformers**.  
It examines images at a deep-learning level to determine whether they are **AI-generated or authentic**, presenting results through a high-contrast cyberpunk forensic interface.

---

## 🚀 Features

- 🧠 AI vs Real image detection using a pretrained diffusion classifier  
- ⚡ Automatic CPU / GPU (CUDA) inference selection  
- 🎯 Confidence score visualization with interactive gauges  
- 🧬 Dynamic, context-aware forensic reasoning  
- 🖼️ Supports JPG, PNG, JPEG, and WEBP images  
- ♻️ Cached model loading for efficient repeated analysis  
- 🔥 Cyberpunk-inspired dark UI and visual effects  

---

## 🧠 Model

**Organika/sdxl-detector**

A pretrained image classification model designed to identify **synthetic diffusion-generated images** by analyzing pixel-level distribution patterns.

---

## 🔬 Forensic Analysis Logic

- Images are normalized and processed using a Hugging Face image processor  
- Model logits are converted into probabilities using softmax  
- Class labels are analyzed to separate **synthetic vs organic signals**  
- A confidence buffer is applied to avoid false positives  
- Context-aware explanations are generated based on score distribution  

---

## 📊 Output Interpretation

- **Verdict**
  - 🚩 AI-GENERATED DETECTED  
  - ✨ AUTHENTIC / ORGANIC  

- **Confidence Score**
  - Percentage-based certainty of the verdict

- **Forensic Reasoning**
  - Human-readable explanation derived from probability behavior

- **Engine Status**
  - Indicates whether inference ran on CPU or GPU

---

## 🧩 Device Intelligence

CyberFury automatically selects the most capable available device:
- CUDA-enabled GPU when available
- CPU fallback otherwise

Model initialization is cached to prevent redundant loading and improve performance.

---

## ⚠️ Disclaimer

CyberFury is a **forensic assistance tool**, not a legal authority.  
Results should be interpreted alongside human expertise and additional forensic evidence.

---

## 📌 Roadmap

- 🔍 Pixel-level artifact heatmaps  
- 🎥 Video deepfake analysis  
- 🌐 API-first deployment mode  
- 📈 Batch forensic scanning  
- 🔐 Tamper-resistant forensic logs  

---

## 👤 Author

**Chirayu Mahajan**  
AI / ML | Digital Forensics | Systems Engineering

---

## ⭐ Acknowledgement

If you find this project valuable, consider supporting it with a ⭐.
