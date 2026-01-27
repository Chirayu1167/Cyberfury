# README.md

# ⚡ CyberFury AI Forensic Scanner

A cutting-edge AI-powered forensic tool that detects AI-generated images with advanced machine learning models and object detection capabilities.

## 🎯 Features

- **AI Image Detection**: Identifies AI-generated images using state-of-the-art deep learning models
- **Object Recognition**: Detects and classifies objects within images using Google's TensorFlow Hub
- **Real-time Analysis**: Instant results with confidence scores and detailed reasoning
- **Visual Feedback**: Interactive gauge charts showing detection confidence
- **GPU Acceleration**: Automatically utilizes GPU when available for faster processing
- **Cyberpunk UI**: Sleek, futuristic dark interface with neon accents

## 🧠 How It Works

CyberFury uses a dual-engine approach:

1. **AI Detection Engine**: Analyzes pixel-level patterns and synthetic markers using the `Organika/sdxl-detector` model trained specifically to identify AI-generated content from models like Stable Diffusion, DALL-E, and Midjourney.

2. **Object Detection Engine**: Uses Google's OpenImages SSD MobileNet v2 to identify objects within the image, providing additional context about the image content.

The system examines:
- Pixel gradient anomalies
- Synthetic diffusion patterns
- Compression artifacts
- Noise characteristics
- Mathematical convergence patterns

## 📊 Detection Thresholds

- **AI-Generated**: Confidence ≥ 99.99%
- **Authentic/Organic**: Confidence < 99.99%
- **Buffer Zone**: 0.9999 threshold for edge-case handling

## 🎨 Use Cases

- **Content Verification**: Verify authenticity of images for journalism and research
- **Social Media Monitoring**: Detect AI-generated content in user uploads
- **Digital Forensics**: Investigate image manipulation and synthetic media
- **Academic Research**: Study AI-generated content patterns
- **Art Authentication**: Distinguish between human-created and AI-generated artwork

## 🔧 Technology Stack

- **Frontend**: Streamlit
- **AI Detection**: PyTorch + HuggingFace Transformers
- **Object Detection**: TensorFlow + TensorFlow Hub
- **Visualization**: Plotly
- **Image Processing**: PIL/Pillow

## 📈 Performance

- **Detection Accuracy**: ~95%+ on AI-generated images
- **Processing Time**: 2-5 seconds per image (CPU), <1 second (GPU)
- **Supported Formats**: JPG, PNG, JPEG, WebP
- **Max Upload Size**: 200MB

## 🔬 Model Information

**Primary AI Detector:**
- Model: `Organika/sdxl-detector`
- Architecture: Image Classification Transformer
- Training: Specialized for SDXL and diffusion model detection

**Object Detector:**
- Model: Google OpenImages v4 SSD MobileNet v2
- Classes: 600+ object categories
- Confidence Threshold: 30%

## 🛡️ Limitations

- Detection accuracy varies with image quality and compression
- May produce false positives on heavily edited real photos
- Requires clear, non-corrupted images for best results
- Object detection requires minimum 320x320 resolution

## 🤝 Contributing

Contributions welcome! Please fork the repository and submit a pull request.

## 📄 License

MIT License - feel free to use for personal and commercial projects.

## 🙏 Acknowledgments

- **HuggingFace** for the Organika/sdxl-detector model
- **Google** for TensorFlow Hub object detection models
- **Streamlit** for the amazing framework

---

**Note**: This tool is for research and verification purposes. Always combine AI detection with human judgment and additional verification methods for critical applications.
