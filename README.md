#HandGaze
Real-Time Hand Gesture Recognition using C++ · OpenCV · TensorFlow · ONNX

🚀 Overview

GestureVision is a real-time computer-vision system that detects and classifies hand gestures using a custom convolutional neural network (CNN).
The model is trained in TensorFlow, exported to ONNX, and deployed in native C++ using OpenCV’s DNN module for fast inference. No Python runtime required.

🧠 Features

Real-time inference (~30 FPS) with OpenCV DNN
Custom CNN trained on palm, fist, and peace gestures
Dynamic ROI detection via skin segmentation & contour tracking
ONNX model integration (TensorFlow → ONNX → OpenCV DNN)
Confidence thresholds and temporal smoothing for stability
Extensible: retrainable with new gestures or datasets

🧰 Tech Stack

Languages: C++, Python
Libraries: OpenCV (4.x), TensorFlow (2.x), tf2onnx, ONNX
Concepts: Computer Vision, CNNs, Model Deployment, Image Processing

⚙️ Setup & Installation

1️⃣ Clone and Build (C++)
git clone https://github.com/yourusername/GestureVision.git
cd GestureVision
mkdir build && cd build
cmake ..
make -j

2️⃣ Train a New Model (Optional)
Activate your Python virtual environment:
python3.11 -m venv tf
source tf/bin/activate
pip install -U pip tensorflow tf2onnx
python train_better.py --epochs 20 --batch 32 --class-weights

This will output:
gestures.onnx
labels.txt

3️⃣ Run the App
./ml_gestures

🎓 Learning Outcomes

Implemented end-to-end ML deployment: Python model → C++ inference
Applied real-time computer vision techniques (masking, contour tracking)
Practiced cross-framework integration and ONNX optimization
Tuned data pipelines and CNN architectures for practical performance

