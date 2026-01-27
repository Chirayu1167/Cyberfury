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
