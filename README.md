# 😷 Real-Time Face Mask Detection (Browser + 3 CNNs)

A production-ready, end-to-end face mask detection system that uses your browser webcam to stream video to a local server, runs inference with **3 CNN models**, and returns real-time ensemble predictions.

## 📦 Features
- 🌐 **Browser Webcam Input**: Secure WebSocket streaming
- 🤖 **3 CNN Models**: MobileNetV2, EfficientNet-B0, Custom Small CNN
- 📊 **Ensemble Predictions**: Averaged softmax probabilities for higher accuracy
- 🖥️ **Live UI Overlay**: Real-time bounding boxes & confidence scores
- 🚀 **One-Click Setup**: Automatic dataset download, training, and server launch
- 💻 **GPU/CPU Support**: Automatic hardware detection

## 🛠️ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt


### 2. run the project 
```bash
python run_all.py