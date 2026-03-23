# 🎓 Classroom Behavior Detection System

> Real-time student behavior detection using YOLOv8x trained on classroom CCTV footage.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8x-Ultralytics-orange)
![mAP](https://img.shields.io/badge/mAP@0.5-74.85%25-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview

This project presents an end-to-end deep learning pipeline for automated visual analysis of student behavior in classroom environments. The system detects five key behaviors in real-time from CCTV footage using YOLOv8x, the largest and most accurate variant of the YOLOv8 family.

Developed as a research project at **Lawrence Technological University** by Darshan Joshi, Harshita Guduru, and Wisam Bukaita, Ph.D.

---

## 🎯 Detected Behaviors

| Class | Description | mAP@0.5 |
|---|---|---|
| ✋ hand_raising | Student raising hand | ~66.5% |
| 📱 using_phone | Student using mobile phone | ~88.8% |
| 😴 sleeping | Student with head down on desk | ~93.5% |
| ✏️ writing | Student writing on paper | ~49.5% |
| 📖 reading | Student reading book or paper | ~71.4% |

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Best mAP@0.5 | **74.85%** (epoch 24) |
| Final mAP@0.5 | 71.44% (epoch 150) |
| mAP@0.5:0.95 | 47.75% |
| Precision | 73.90% |
| Recall | 67.08% |

Training: 150 epochs on Google Colab Pro (NVIDIA L4 GPU, 23.7GB VRAM)

---

## 🗂️ Dataset

- **Source:** [Roboflow Student Behaviour Detection](https://universe.roboflow.com/mywork-lkwz4/student-behaviour-detection-neazg) (CC BY 4.0)
- **Version:** 6
- **Images:** 3,104 train / 483 val / 279 test
- **Classes:** 5 (remapped from original 12)

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/classroom-behavior-detection
cd classroom-behavior-detection
pip install -r requirements.txt
```

### Run Inference on a Video
```bash
python inference.py classroom.mp4 best.pt output.mp4
```

### Run Inference on an Image
```python
from ultralytics import YOLO

model = YOLO('best.pt')
results = model('classroom.jpg', conf=0.35)
results[0].show()
```

### Tiled Inference (for wide-angle CCTV)
```python
from inference import run_on_video

run_on_video(
    video_path='classroom.mp4',
    model_path='best.pt',
    output_path='output_annotated.mp4',
    skip=5
)
```

---

## 📁 Repository Structure

```
classroom-behavior-detection/
├── inference.py              # Tiled inference script for video
├── requirements.txt          # Python dependencies
├── dataset.yaml              # Dataset configuration
├── results.csv               # Training metrics per epoch
├── charts/
│   ├── results.png           # Training curves
│   ├── confusion_matrix_normalized.png
│   ├── BoxF1_curve.png
│   ├── BoxPR_curve.png
│   ├── val_batch0_pred.jpg   # Validation predictions
│   └── val_batch1_pred.jpg
└── README.md
```

---

## 🏗️ Architecture

- **Model:** YOLOv8x (68.1M parameters, 258.1 GFLOPs)
- **Backbone:** C2f modules for hierarchical feature extraction
- **Neck:** PANet for multi-scale feature fusion
- **Head:** Decoupled detection head
- **Input size:** 640×640
- **Optimizer:** SGD (lr=0.01, cosine decay)

---

## 🎥 Real-World Deployment

For wide-angle CCTV deployment, a **tiled inference** approach is used:

1. Each frame is split into overlapping 640×640 tiles
2. Inference runs on each tile independently
3. Results are mapped back to full-frame coordinates
4. NMS removes duplicate detections at tile boundaries

This improves detection coverage from ~2 students per frame to **15-25 students per frame** on wide-angle footage.

---

## 📈 Training Results

![Training Results](charts/results.png)

![Confusion Matrix](charts/confusion_matrix_normalized.png)

---

## 🔮 Future Work

- Fine-tune on target classroom data to address domain shift
- Temporal tracking for behavioral trend analysis
- Edge deployment optimization (TensorRT, ONNX)
- Multi-classroom dashboard with alerts

---

## 📄 Citation

```bibtex
@misc{joshi2026classroom,
  title={Automated Visual Analysis of Classroom Behavior Using Deep Learning},
  author={Joshi, Darshan and Guduru, Harshita and Bukaita, Wisam},
  institution={Lawrence Technological University},
  year={2026}
}
```

---

## 📜 License

This project is licensed under the MIT License.

Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — Roboflow Student Behaviour Detection by Mywork.
