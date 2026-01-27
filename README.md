# ⚡ CyberFury: AI Forensic Scanner

An AI-powered forensic tool to detect AI-generated images using state-of-the-art machine learning models.

## 🚀 Features

- **AI Detection**: Detects AI-generated images with high accuracy using HuggingFace transformers
- **Object Detection**: Identifies objects in images using Google's TensorFlow Hub models
- **Real-time Analysis**: Instant results with confidence scores
- **GPU Support**: Automatically uses GPU if available for faster processing

## 📦 Installation

### Local Setup

```bash
# Clone the repository
git clone <https://github.com/Chirayu1167/Cyberfury>
cd cyberfury

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 🌐 Deployment on Streamlit Cloud

### ⚠️ Known Issue: Network Restrictions

Streamlit Cloud has network restrictions that prevent downloading models from HuggingFace at runtime.

### Solution Options:

#### **Option 1: Pre-download Models (Recommended)**

1. Run this locally to download models:

```python
# download_models.py
from transformers import AutoImageProcessor, AutoModelForImageClassification

model_name = "Organika/sdxl-detector"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

processor.save_pretrained("./models/sdxl-detector")
model.save_pretrained("./models/sdxl-detector")
print("✅ Models saved to ./models/sdxl-detector")
```

2. Run: `python download_models.py`

3. Commit the `models/` folder to your repository:
```bash
git add models/
git commit -m "Add pre-downloaded models"
git push
```

4. The app will automatically use local models if available!

#### **Option 2: Use Git LFS for Large Files**

```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "models/**/*.bin"
git lfs track "models/**/*.safetensors"

# Commit and push
git add .gitattributes
git add models/
git commit -m "Add models with Git LFS"
git push
```

#### **Option 3: Deploy on Alternative Platforms**

These platforms have unrestricted network access:

- **Hugging Face Spaces** (Free, best for ML apps)
- **Railway.app** (Free tier available)
- **Render.com** (Free tier available)
- **Heroku** (Paid plans)

### Deployment on Hugging Face Spaces (Recommended Alternative)

1. Create account at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Create new Space → Select "Streamlit"
3. Upload `app.py` and `requirements.txt`
4. Models download automatically (no restrictions!)

## 🔧 Troubleshooting

### Error: "OSError: This app has encountered an error"

**Cause**: Network restrictions preventing model downloads

**Solution**: Use pre-downloaded models (see Option 1 above)

### Error: "Google detector loading issue"

**Cause**: TensorFlow Hub connection issues

**Fix**: This is non-critical. AI detection will still work, only object detection is affected.

### Models take too long to load

**Solution**: 
- Use pre-downloaded models
- Deploy on platforms with better resources
- Consider using smaller models

## 📊 Model Information

- **AI Detector**: `Organika/sdxl-detector` (HuggingFace)
- **Object Detector**: Google OpenImages v4 SSD MobileNet v2 (TensorFlow Hub)

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI Detection**: PyTorch + Transformers
- **Object Detection**: TensorFlow + TensorFlow Hub
- **Visualization**: Plotly

## 📝 File Structure

```
cyberfury/
├── app.py                  # Main application
├── requirements.txt        # Python dependencies
├── packages.txt           # System dependencies (for Streamlit Cloud)
├── README.md              # This file
└── models/               # Pre-downloaded models (optional)
    └── sdxl-detector/
        ├── config.json
        ├── model.safetensors
        └── ...
```

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📄 License

MIT License

## 🙏 Credits

- Model: [Organika/sdxl-detector](https://huggingface.co/Organika/sdxl-detector)
- Object Detection: Google TensorFlow Hub
- Framework: Streamlit

---

**Note**: This app works perfectly locally. Deployment issues are solely due to Streamlit Cloud's network restrictions. Use pre-downloaded models or alternative platforms for cloud deployment.
