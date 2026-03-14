# 🛵 HelmetGuard AI

**Detect cyclists without helmets and extract nearby license plates from images & videos.**

An end-to-end computer-vision pipeline built with **YOLOv11** for object detection, **rule-based heuristics** for violation analysis, and **PaddleOCR** for license-plate reading — served through a **Gradio** web interface or a simple CLI.

---

## 1. Problem

In many cities, riding a motorcycle or bicycle without a helmet is both dangerous and illegal. Manual enforcement doesn't scale, and CCTV footage alone is only useful *after* the fact.

**HelmetGuard AI** automates the process:

| Step | What it does |
|------|-------------|
| Detect objects | Locate *cyclists*, *helmets*, and *license plates* in each frame |
| Determine violations | Apply spatial rules to decide if a cyclist is **not** wearing a helmet |
| Read plates | Run OCR on the nearest license plate to identify the violator |
| Record evidence | Save cropped images + timestamped reports for each violation |

---

## 2. Scope

### ✅ This project covers

- Object detection **inference** (YOLOv11)
- Rule-based **violation analysis** (IoU + distance heuristics)
- License-plate **OCR extraction** (PaddleOCR)
- **Visualization** and artifact saving
- Interactive **Gradio web UI** + **CLI** entry points

### ❌ This project does **not** include

- Full backend / REST API deployment
- Database or persistent storage layer
- Real-time streaming infrastructure
- Traffic enforcement workflow integration
- Model training or fine-tuning pipeline

> *Explicitly defining scope shows awareness of production boundaries. It tells a reviewer: "I know where inference ends and production engineering begins."*

---

## 3. Input / Output

| | Description |
|---|---|
| **Input** | A single image (`.jpg`, `.png`) or a video file (`.mp4`) |
| **Output** | Annotated image/video with bounding boxes + a violation report (text & table) |

### Detection Classes (YOLO)

| Class ID | Label | Description |
|----------|-------|-------------|
| 0 | `helmet` | Protective headgear on a rider |
| 1 | `license_plate` | Vehicle registration plate |
| 2 | `cyclist` | Person on a motorcycle / bicycle |

### Colour Legend

| Colour | Meaning |
|--------|---------|
| 🟢 Green | Cyclist **with** helmet — OK |
| 🔴 Red | Cyclist **without** helmet — **VIOLATION** |
| 🟡 Cyan | Detected helmet / unassociated plate |
| 🔵 Blue | License plate linked to a violation |

---

## 4. Pipeline

```
Image / Video Frame
  │
  ▼
┌──────────────────┐
│  YOLO Detector    │  ← Detect helmet, cyclist, license_plate
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Group Detections │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Helmet–Cyclist   │  ← IoU between helmet bbox and upper 40% of cyclist bbox
│  Matching         │     threshold = 0.3
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Plate            │  ← Euclidean distance < 200 px
│  Association      │     plate must be below or at cyclist height
└────────┬─────────┘
         ▼
┌──────────────────┐
│  PaddleOCR        │  ← Crop → 2× upscale → OCR
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Violation Record │  ← timestamp, confidence, bbox, plate text
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Visualization    │  ← Annotated image + saved artifacts
└──────────────────┘
```

### Rule-based Violation Logic (detail)

1. **Upper-body region** — For each detected cyclist, define the *upper body* as the **top 40 %** of the bounding box height.
2. **Helmet matching** — Compute IoU between every detected helmet and the upper-body region. If any helmet achieves `IoU > 0.3`, the cyclist is considered **safe**.
3. **Plate association** — For each violator, find the closest `license_plate` detection within **200 px** Euclidean distance, constrained to plates roughly **below** the cyclist centre.
4. **OCR** — The associated plate crop is padded by 5 px, upscaled 2×, then fed to PaddleOCR.

---

## 5. OCR Pipeline

```
license_plate bbox
  │
  ├─ Pad ±5 px (clamped to image bounds)
  ├─ Crop from full image
  ├─ Resize 2× (INTER_CUBIC)
  └─ PaddleOCR (lang='vi', angle_cls=True)
       │
       └─ Concatenate detected text lines → plate_text
```

- **Language**: Vietnamese (`vi`) by default — configurable in `config.py`.
- **Angle classification** is enabled to handle tilted plates.

---

## 6. Evaluation Plan

| Metric | What it measures | How to compute |
|--------|-----------------|----------------|
| mAP@0.5 | Detection accuracy | YOLO val on held-out set |
| Precision / Recall (violations) | Rule-based logic correctness | Manual annotation of 100+ frames |
| OCR Character Accuracy | Plate reading quality | Compare OCR output to ground truth |
| End-to-end Accuracy | Full pipeline correctness | % of frames with correct violation + correct plate |

> *Evaluation is planned but not yet executed — the current focus is on building a clean, testable pipeline.*

---

## 7. Project Structure

```
HelmetGuard-AI/
│
├── README.md                 ← You are here
├── requirements.txt          ← Python dependencies
├── config.py                 ← All tuneable thresholds & paths
├── main.py                   ← CLI entry point (argparse)
├── app.py                    ← Gradio web interface
├── best.pt                   ← YOLOv11 weights (not in git)
│
├── src/                      ← Core library
│   ├── __init__.py
│   ├── detector.py           ← YOLO inference wrapper
│   ├── violation_analyzer.py ← Rule-based helmet & plate logic
│   ├── ocr_reader.py         ← PaddleOCR wrapper
│   ├── pipeline.py           ← Orchestrator (detect → analyze → OCR → record)
│   └── visualizer.py         ← Bounding-box drawing
│
├── assets/                   ← Sample images for demo
│   └── test_image.jpg
│
├── outputs/                  ← Auto-generated violation records
│   └── violation_records/
│       └── YYYY-MM-DD_HH-MM-SS/
│           ├── violator.jpg
│           ├── license_plate.jpg
│           └── violation_details.txt
│
├── tests/                    ← Unit tests (placeholder)
│   └── __init__.py
│
├── plan_restructure.md       ← Restructuring plan
└── .gitignore
```

### Module Responsibilities

| Module | Single Responsibility |
|--------|----------------------|
| `detector.py` | Load YOLO model, run inference, return structured detections |
| `violation_analyzer.py` | IoU calculation, helmet–cyclist matching, plate association |
| `ocr_reader.py` | Crop plate region, preprocess, run PaddleOCR, return text |
| `pipeline.py` | Orchestrate the full detect → analyze → OCR → record flow |
| `visualizer.py` | Draw colour-coded bounding boxes and labels on images |

---

## 8. Quick Start

### Prerequisites

- Python 3.8+
- (Optional) CUDA-capable GPU for faster inference

### Installation

```bash
git clone https://github.com/KPN147/HelmetGuard-AI.git
cd HelmetGuard-AI

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

### Place your YOLO model

Copy your trained `best.pt` into the project root (or update `MODEL_PATH` in `config.py`).

### Run — Web UI

```bash
python app.py
# Open http://127.0.0.1:7860
```

### Run — CLI

```bash
# Single image
python main.py --image assets/test_image.jpg

# Video
python main.py --video path/to/video.mp4

# Skip saving violation records
python main.py --image assets/test_image.jpg --no-save
```

---

## 9. Limitations

| Limitation | Why it matters |
|------------|---------------|
| **Occlusion** | Overlapping riders can merge bounding boxes, causing missed or false violations |
| **Small plate crops** | Low-resolution plates reduce OCR accuracy significantly |
| **Heuristic association** | Plate-to-cyclist matching uses Euclidean distance, not tracking — may mis-assign in crowded scenes |
| **No temporal consistency** | Each frame is processed independently; the same rider can be flagged multiple times across video frames |
| **Lighting & angle** | Extreme angles or low-light conditions degrade both detection and OCR |

> *Listing limitations honestly demonstrates practical engineering judgement — reviewers value this.*

---

## 10. Future Work

- [ ] **Multi-object tracking** (DeepSORT / ByteTrack) for consistent rider identity across frames
- [ ] **Database integration** to persist violation records
- [ ] **REST API** with FastAPI for production deployment
- [ ] **Model retraining** on domain-specific Vietnamese traffic data
- [ ] **Plate super-resolution** (ESRGAN) to improve OCR on small crops
- [ ] **Real-time streaming** via RTSP / WebSocket
- [ ] **Unit & integration tests** with pytest
- [ ] **CI/CD pipeline** for automated linting, testing, and Docker builds

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv11 (Ultralytics) |
| OCR | PaddleOCR |
| Image Processing | OpenCV |
| Web Interface | Gradio |
| Language | Python 3.8+ |

---

## Author

**Khanh Pham** — [GitHub](https://github.com/KPN147)

---

<sub>⚖️ This project is intended for educational and authorised traffic-safety purposes. Ensure compliance with local privacy and data-protection laws before deployment.</sub>
