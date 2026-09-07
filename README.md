<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f172a,50:1e3a8a,100:0ea5e9&height=220&section=header&text=Safe%20Ride%20Vision&fontSize=55&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=AI-Powered%20Rider%20Safety%20%26%20Traffic%20Violation%20Detection&descAlignY=58&descSize=18" width="100%"/>

<br/>

<a href="#">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=24&duration=2600&pause=900&color=0EA5E9&center=true&vCenter=true&multiline=true&repeat=true&width=800&height=90&lines=Detecting+bikes+with+YOLO+%2B+DeepSORT+%F0%9F%9A%B2;Reading+number+plates+with+OCR+%F0%9F%94%A4;Catching+drowsy+blinks+with+CNN-LSTM+%F0%9F%91%81%EF%B8%8F;Flagging+illegal+turns+at+junctions+%F0%9F%9A%A6" alt="Typing SVG" />
</a>

<br/><br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-DL-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-CV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Backend-000000?style=for-the-badge&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge)

<img src="https://img.shields.io/github/stars/your-org/safe-ride-vision?style=social" />
<img src="https://img.shields.io/github/forks/your-org/safe-ride-vision?style=social" />

</div>

<br/>

## 🎯 What is Safe Ride Vision?

> A real-time computer-vision pipeline that watches traffic footage, tracks every motorbike frame-by-frame, reads number plates automatically, detects drowsy/blinking riders, and flags illegal turns at junctions — all packaged behind a simple video-upload API.

<div align="center">
<table>
<tr>
<td align="center" width="20%">🏍️<br/><b>Multi-Object Tracking</b><br/><sub>DeepSORT + Kalman Filter</sub></td>
<td align="center" width="20%">🔢<br/><b>Plate Recognition</b><br/><sub>YOLO detector + PaddleOCR</sub></td>
<td align="center" width="20%">😴<br/><b>Drowsiness Detection</b><br/><sub>CNN + LSTM/GRU/Transformer</sub></td>
<td align="center" width="20%">🚦<br/><b>Turn Violation Logic</b><br/><sub>Angle + Zone detection</sub></td>
<td align="center" width="20%">☁️<br/><b>Cloud-Ready Backend</b><br/><sub>Flask + ngrok / Colab</sub></td>
</tr>
</table>
</div>

<br/>

<div align="center">
<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.gif" width="100%" height="4px"/>
</div>

## 🧠 Architecture

```mermaid
flowchart LR
    A[📹 Uploaded Video] --> B[YOLO Detector]
    B --> C[DeepSORT Tracker]
    C --> D{Per Bike Track}
    D --> E[Plate Detector - YOLO]
    E --> F[PaddleOCR - end of run]
    D --> G[Blink / Drowsiness Model]
    G --> G1[CNN Backbone]
    G1 --> G2[LSTM / GRU / BiLSTM / Transformer]
    G2 --> G3[Classification Head]
    D --> H[Turn / Junction Logic]
    H --> H1[Angle Detector]
    H --> H2[Zone Overlap Check]
    F --> I[📊 Annotated Video + CSV Logs]
    G3 --> I
    H1 --> I
    H2 --> I
    I --> J[🌐 Flask API Response]

    style A fill:#0ea5e9,color:#fff
    style J fill:#1e3a8a,color:#fff
    style I fill:#0f172a,color:#fff
```

<br/>

## ✨ Feature Breakdown

<details open>
<summary><b>🏍️ Tracking Engine — <code>track.py</code> / <code>deepsortVideo.py</code></b></summary>
<br/>

- Classic **DeepSORT** track lifecycle: `Tentative → Confirmed → Deleted`
- Kalman-filter based motion prediction with `(x, y, aspect_ratio, height)` state space
- Robust exception logging per track ID so silent crashes never go unnoticed
- Full per-video pipeline: detection → tracking → plate crop → blink check → turn check → CSV export

</details>

<details>
<summary><b>🔢 Plate Detection & OCR — <code>plate_detector.py</code></b></summary>
<br/>

- **Split-stage design**: plate *detection* (YOLO) runs every frame; **OCR** (PaddleOCR) runs once at the end on cropped plates only — huge speed win
- `debug=True` mode streams live intermediate images (Colab-safe via `cv2_imshow`, matplotlib fallback elsewhere)
- Threading watchdog warns if any OCR call hangs — guards against PaddlePaddle's known CPU-inference deadlock
- `warm_up_plate_detector()` / `warm_up_ocr()` for zero cold-start latency mid-run

</details>

<details>
<summary><b>😴 Drowsiness / Blink Classifier — <code>blink_model.py</code></b></summary>
<br/>

A fully modular **CNN → Sequence Model → FC Head** hybrid:

| Component | Options |
|---|---|
| CNN Backbone | Any `torchvision.models` classifier (ResNet18 default) |
| Sequence Head | `LSTM` · `GRU` · `BiLSTM` · `RNN` · `Transformer` |
| Config | Single `ModelConfig` dataclass — swap architectures with one line |

```python
from blink_model import build_model, ModelConfig

cfg = ModelConfig(cnn_name="resnet18", seq_type="LSTM",
                   num_layers=2, hidden_size=256, dropout=0.5)
model = build_model(cfg)
logits = model(video_batch, lengths)
```

</details>

<details>
<summary><b>🚦 Junction & Turn Violation Logic — <code>backend_server.py</code></b></summary>
<br/>

- Two operating modes: **`direct`** (angle-detector only) and **`junction`** (angle detector **OR** polygon zone overlap)
- Frontend-drawn rectangles / circles / polygons are rasterized into `TURN_ZONES` automatically
- Every processed video gets its **own output folder** — re-uploading the same filename cleanly overwrites instead of piling up
- Per-bike snapshots (`annotated_frame.jpg`, `plate_crop.jpg`) served directly over static routes

</details>

<br/>

## 📁 Project Structure

```
safe-ride-vision/
├── 🎥 deepsortVideo.py      # Main pipeline orchestrator (detect → track → analyze)
├── 🧭 track.py              # DeepSORT track state machine
├── 🔢 plate_detector.py     # Plate YOLO detector + PaddleOCR module
├── 👁️  blink_model.py        # CNN + Sequence hybrid drowsiness classifier
├── 🌐 backend_server.py     # Flask + ngrok inference API
└── 📓 *.ipynb               # Colab notebook entry point
```

<br/>

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/your-org/safe-ride-vision.git
cd safe-ride-vision

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the backend (Colab or local GPU box)
python backend_server.py
```

**API Contract**

```http
POST /process-video
Content-Type: multipart/form-data

video            → the video file
mode             → "direct" | "junction"
fps              → float            (junction mode only)
junction_count   → int              (junction mode only)
junctions        → JSON array       (junction mode only)
```

```json
{
  "output_video_url": "/outputs/<file>.mp4",
  "logs": [ { "frame_idx": 0, "track_id": 1, "cx": 320, "cy": 180 } ]
}
```

<br/>

## 🛠️ Tech Stack

<div align="center">

![PyTorch](https://skillicons.dev/icons?i=pytorch)
![Python](https://skillicons.dev/icons?i=python)
![OpenCV](https://skillicons.dev/icons?i=opencv)
![Flask](https://skillicons.dev/icons?i=flask)
![Docker](https://skillicons.dev/icons?i=docker)

</div>

<br/>

## 🗺️ Roadmap

- [x] Real-time multi-object tracking with DeepSORT
- [x] Plate detection + delayed-batch OCR for speed
- [x] Modular blink/drowsiness classifier with swappable sequence heads
- [x] Junction zone + angle-based turn violation detection
- [ ] Live RTSP stream support
- [ ] Edge-device (Jetson) deployment
- [ ] Web dashboard for live analytics

<br/>

<div align="center">

### 💙 Built for safer roads

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0ea5e9,100:0f172a&height=120&section=footer" width="100%"/>

</div>
