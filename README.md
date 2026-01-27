# ⚡ CyberFury | AI Forensic Lab

**CyberFury** is a high-performance digital forensics tool designed to distinguish between authentic photography and AI-generated imagery. By combining deep learning pixel analysis with exhaustive metadata inspection, it provides a comprehensive verdict on the legitimacy of digital evidence.

---

## 🚀 Core Features

### 1. Neural Pixel Analysis
The engine utilizes the `Organika/sdxl-detector` transformer model to scan for "synthetic fingerprints"—microscopic patterns left behind by diffusion models like Stable Diffusion XL. It provides a real-time confidence score for every scan.



### 2. EXIF Forensic Suite
The tool deep-dives into the image header data to uncover:
* **AI Software Signatures:** Detects traces of Midjourney, DALL-E, ComfyUI, and more.
* **Camera Integrity:** Identifies missing "Make" and "Model" tags common in generated images.
* **Format Discrepancies:** Flags suspicious WebP or PNG files that lack standard capture metadata.

### 3. Integrated Fact-Checking Bridge
To ensure a 360-degree investigation, CyberFury provides direct portals to:
* **Filename Source Search:** Automatically queries Google for specific file naming conventions used by AI generators.
* **Google Fact-Check Database:** Cross-references the image against known debunked visual claims.
* **Lens Integration:** A streamlined workflow for reverse-image searching to find the original context of a file.

---

## 🛠 How It Works

1.  **Upload:** Provide a JPG, PNG, or WEBP file for analysis.
2.  **Execute:** The `CyberFuryEngine` processes the image through a GPU/CPU-accelerated neural network.
3.  **Metadata Scrub:** The system extracts and parses EXIF tags, looking for "Suspicious Indicators."
4.  **Verdict:** The UI generates a high-contrast report:
    * 🔵 **REAL:** High confidence that the image was captured by a physical camera.
    * 🔴 **AI:** High probability of synthetic generation.

---

## 📡 Technical Architecture

* **Frontend:** Streamlit (Cyberpunk-themed UI)
* **ML Engine:** Hugging Face Transformers (`AutoModelForImageClassification`)
* **Image Processing:** PIL (Pillow)
* **Inference:** PyTorch (Support for CUDA acceleration)

---
By: Chirayu Mahajan


## ⚖ Disclaimer
*Forensic analysis is probabilistic. While CyberFury uses state-of-the-art detection models, it should be used as one part of a broader investigative process. Highly edited real photos or heavily compressed AI images may occasionally yield false positives/negatives.*
