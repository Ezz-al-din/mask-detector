# 😷 Real-Time Face Mask Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![3-Model Ensemble](https://img.shields.io/badge/Architecture-3--Model%20Ensemble-orange)]()

A production-ready, browser-based face mask detection system that runs **3 CNN models in parallel** for real-time ensemble predictions via your webcam.

---

## 🌟 Features

- 🎥 **Live Browser Webcam** - Secure WebSocket streaming with real-time overlay
- 🤖 **3-Model Ensemble** - Combines predictions from MobileNetV2, EfficientNet-B0, and Custom CNN
- 📊 **High Accuracy** - Ensemble averaging improves robustness over single models
- ⚡ **Fast Inference** - ~15-20 FPS on CPU, ~30+ FPS on GPU
- 🔒 **Privacy-First** - All processing happens locally on your machine
- 🚀 **One-Click Setup** - Automatic dataset preparation, training, and deployment

---

## 🏗️ System Architecture

### 🔹 3 CNN Models (Ensemble)
| Model | Parameters | Speed | Role |
|-------|-----------|-------|------|
| **MobileNetV2** | ~3.5M | ⚡⚡ Fast | Balanced speed/accuracy |
| **EfficientNet-B0** | ~5.3M | ⚡ Medium | High accuracy backbone |
| **Custom CNN** | ~0.5M | ⚡⚡⚡ Very Fast | Lightweight fallback |

### 🔹 2-Class Classification
- **Class 0**: `Mask` (person wearing face mask correctly)
- **Class 1**: `No Mask` (person without mask or incorrect usage)

### 🔹 Ensemble Prediction Flow

Camera Frame → Center Crop → Resize to 224x224
↓
┌──────────────┬──────────────┬──────────────┐
│ MobileNetV2 │ EfficientNet │ Custom CNN │
└──────┬───────┴──────┬───────┴──────┬───────┘
↓ ↓ ↓
[Softmax Probs] [Softmax Probs] [Softmax Probs]
└──────┬───────┴──────┬───────┘
↓ AVERAGE ↓
Final Prediction → Bounding Box Overlay


---

## 📁 Project Structure

mask-detector/
├── README.md # This documentation
├── requirements.txt # Python dependencies
├── run_all.py # 🔥 One-click: Prepare → Train → Serve
├── download_data.py # Prepares local dataset (80/20 split)
├── train_models.py # Trains 3 CNNs & saves weights
├── server.py # Flask-SocketIO inference server
│
├── data/ # Dataset directory (you provide)
│ ├── with_mask/ # Raw mask images (input)
│ ├── without_mask/ # Raw no-mask images (input)
│ ├── train/ # Auto-generated 80% split
│ │ ├── mask/
│ │ └── no_mask/
│ └── test/ # Auto-generated 20% split
│ ├── mask/
│ └── no_mask/
│
├── models/ # Trained weights (auto-generated)
│ ├── mobilenet_v2_mask.pth
│ ├── efficientnet_b0_mask.pth
│ └── custom_cnn_mask.pth
│
├── templates/
│ └── index.html # Web UI with webcam
└── static/
├── css/style.css # Styling
└── js/app.js # WebSocket client & overlay

---

## 🚀 Quick Start

### 1️⃣ Clone & Install
```bash
git clone https://github.com/yourusername/mask-detector.git
cd mask-detector
pip install -r requirements.txt
```

## 2️⃣ Prepare Your Dataset

Place your images if you want to use another dataset in the `data/` folder exactly like this:

data/
├── with_mask/      # Add images of people wearing masks
└── without_mask/   # Add images of people without masks


**Supported formats:** `.jpg`, `.jpeg`, `.png`  
**Recommended:** 500+ images per class for reliable accuracy  

---

## 3️⃣ Run Everything (One Command)

```bash
python run_all.py
```
### This script will:

- ✅ **Prepare Dataset** – Shuffle, split 80/20 train/test, rename folders  
- 🧠 **Train 3 Models** – MobileNetV2, EfficientNet-B0, Custom CNN (~10–15 min on CPU)  
- 💾 **Save Weights** – Automatically saves `.pth` files to `models/`  
- 🌐 **Start Server** – Launch Flask-SocketIO at http://localhost:5000  

---

## 4️⃣ Open in Browser

Navigate to http://localhost:5000, click **"Start Camera"**, and allow webcam access.

---

## 📂 Dataset Preparation Details

The `download_data.py` script does **not** download anything. It expects local folders and handles:

- ✅ Validates `data/with_mask/` and `data/without_mask/` exist  
- ✅ Shuffles images (`seed=42` for reproducibility)  
- ✅ Splits 80% → `data/train/`, 20% → `data/test/`  
- ✅ Renames classes to `mask` and `no_mask` (required for PyTorch `ImageFolder`)  
- ✅ Leaves original folders untouched (non-destructive)  

---

## 📁 Expected Output Structure

data/
├── with_mask/        ← Your original images (kept)
├── without_mask/     ← Your original images (kept)
├── train/
│   ├── mask/         [80% of mask images]
│   └── no_mask/      [80% of no-mask images]
└── test/
    ├── mask/         [20% of mask images]
    └── no_mask/      [20% of no-mask images]

---
## 🧠 Training Details

### ⚙️ Configuration

| Parameter        | Value                                      |
|------------------|--------------------------------------------|
| Epochs           | 10                                         |
| Batch Size       | 32                                         |
| Optimizer        | Adam (lr = 0.001)                          |
| Scheduler        | StepLR (decay 0.1 every 5 epochs)          |
| Loss             | CrossEntropyLoss                           |
| Input Size       | 224 × 224                                  |
| Normalization    | ImageNet stats `[0.485, 0.456, 0.406]`     |

---

### 🔄 Data Augmentation (Train Only)

- Random resized crop  
- Random horizontal flip  
- Color jitter (brightness & contrast ±20%)  
- `.convert('RGB')` to fix PNG transparency/palette warnings  

---

### 💾 Model Saving

All 3 models are automatically saved after training completes: