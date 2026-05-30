# YOLO CV

Object detection scripts using YOLOv8 — static images and live webcam feed, with custom OpenCV rendering.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## Overview

Three focused Python scripts that demonstrate YOLOv8-based object detection via the Ultralytics library. Covers three distinct workflows: Ultralytics' built-in visualization pipeline, manual bounding box rendering with OpenCV, and real-time webcam inference. Aimed at engineers learning the difference between high-level model APIs and low-level frame manipulation.

## Highlights

- **Two rendering paths side by side** — `yolo.py` uses Ultralytics' `results.plot()` for zero-config output; `yolo2.py` bypasses it entirely, drawing boxes and labels directly with `cv2.rectangle` and `cv2.putText`, making the detection data structure explicit
- **Real-time inference at 1280x720** — `yolo3.py` captures webcam frames at HD resolution and runs YOLOv8m through the same per-frame loop, demonstrating a complete inference loop with graceful teardown on `q`
- **Model weight progression** — static scripts use `yolov8s` (small, faster); live detection upgrades to `yolov8m` (medium, higher accuracy), illustrating the speed/accuracy tradeoff in practice
- **No custom training required** — all three scripts run against the COCO-pretrained weights that Ultralytics auto-downloads on first run

## Features

**Static Image Detection**
- Run `yolov8s.pt` against any JPEG input
- Two output modes: Ultralytics-rendered (`output.jpg`) and manually drawn bounding boxes (`output2.jpg`)
- Confidence scores displayed alongside class names

**Live Webcam Detection**
- Frame-by-frame inference using `yolov8m.pt`
- 1280x720 capture resolution configurable via `cap.set`
- Real-time annotated display window; press `q` to exit cleanly

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Inference | Ultralytics YOLOv8 | Object detection model and pretrained weights |
| Vision | OpenCV (`opencv-python`) | Image I/O, frame capture, drawing primitives |
| Data | NumPy | Frame array manipulation in webcam loop |
| Runtime | Python 3.8+ | Script execution environment |

## How It Works

1. **Model load** — `YOLO('yolov8s.pt')` or `YOLO('yolov8m.pt')` initializes the model; weights are downloaded automatically from Ultralytics on the first call if not cached locally.
2. **Input acquisition** — static scripts read a JPEG with `cv2.imread`; the webcam script opens device `0` at 1280x720 and reads frames in a `while True` loop.
3. **Inference** — `model(img)[0]` or `model(frame)[0]` runs a single forward pass and returns a `Results` object containing `boxes.data` (a tensor of `[x1, y1, x2, y2, score, class_id]` rows).
4. **Rendering** — either `results.plot()` for automatic annotation, or a manual loop over `boxes.data.tolist()` that calls `cv2.rectangle` and `cv2.putText` with explicit coordinates and colors.
5. **Output** — static scripts write the annotated frame to disk with `cv2.imwrite`; the webcam script displays the live window and releases the capture device on exit.

## Setup

**Prerequisites:** Python 3.8+

```bash
# Install dependencies
pip install -r requirements.txt
```

Dependencies installed: `ultralytics`, `opencv-python`, `mediapipe`

Model weights (`yolov8s.pt`, `yolov8m.pt`) are downloaded automatically by Ultralytics on first run — no manual download needed.

## Usage

**Static detection with built-in visualization**
```bash
python yolo.py
# Reads test.jpg, writes annotated output to output.jpg
```

**Static detection with manual bounding boxes**
```bash
python yolo2.py
# Reads test.jpg, draws custom boxes/labels, writes output2.jpg
```

**Real-time webcam detection**
```bash
python yolo3.py
# Opens webcam at 1280x720, displays live annotated feed
# Press q to quit
```

## Key Decisions

| Decision | Rationale | Tradeoff |
|---|---|---|
| Use `results.plot()` in `yolo.py` vs manual drawing in `yolo2.py` | Demonstrates both the high-level API and the underlying data structure in isolation | `results.plot()` is simpler but opaque; manual drawing requires parsing `boxes.data` directly |
| `yolov8s` for static, `yolov8m` for webcam | Static scripts prioritize speed and simplicity; webcam mode benefits from the accuracy gain of the medium model | `yolov8m` is slower per frame — on CPU hardware this will reduce live framerate |
| `mediapipe` in requirements but not used in current scripts | Dependency present for future hand/pose landmark extensions | Adds install weight without immediate value |

## About

Built to explore the gap between using a model library's convenience API and understanding what it actually produces. Working through both rendering approaches — `results.plot()` vs raw `boxes.data` — makes the inference output concrete rather than a black box, which is useful grounding before tackling custom pipelines or fine-tuning workflows.
