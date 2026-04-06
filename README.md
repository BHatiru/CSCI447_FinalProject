# CSCI 447 Final Project — SAR Oil Spill Segmentation with YOLOv8

Instance segmentation of oil spills in Synthetic Aperture Radar (SAR) satellite imagery using YOLOv8. The model detects and segments four classes of interest in radar images: **Oil Spill**, **Look-alike**, **Ship**, and **Land**.

## Problem Statement

Oil spills in oceans pose severe environmental and ecological threats. SAR imagery enables all-weather, day-and-night monitoring of ocean surfaces, making it a practical tool for oil spill detection. However, distinguishing genuine oil spills from look-alike phenomena (e.g., low-wind zones, biogenic films) remains challenging. This project applies deep learning–based instance segmentation to automate and improve this classification.

## Dataset

The dataset consists of **1250×650 pixel SAR images** with corresponding color-coded segmentation masks.

| Class | Mask Color (BGR) | Description |
|-------|-------------------|-------------|
| Oil Spill | (255, 255, 0) Cyan | Actual oil spill regions |
| Look-alike | (0, 0, 255) Red | False positives — natural ocean phenomena |
| Ship | (0, 76, 153) Brown | Vessel detections |
| Land | (0, 153, 0) Green | Land masses |

The data is split into **train**, **validation**, and **test** sets. Raw mask images are converted through a two-stage pipeline into YOLO-compatible annotations.

## Data Pipeline

```
Raw Masks (RGB)  →  COCO JSON  →  YOLO TXT
   (masks/)       masks_to_coco.py   coco_to_yolo.py
```

1. **`utils/train_val_split.py`** — Splits the raw dataset into 80% train / 20% validation with image–mask correspondence preserved.
2. **`utils/masks_to_coco.py`** — Converts color-coded mask images into COCO JSON format by extracting contours, computing bounding boxes, areas, and segmentation polygons for each class.
3. **`utils/coco_to_yolo.py`** — Converts COCO JSON annotations into YOLO format (normalized polygon coordinates in `.txt` label files) and generates the `data.yaml` configuration.

### Annotation Formats

**COCO JSON** — Standard COCO structure with `images`, `categories`, and `annotations` arrays containing bounding boxes and segmentation polygons.

**YOLO TXT** — One file per image. Each line: `<class_id> <x1> <y1> <x2> <y2> ...` with coordinates normalized to [0, 1].

## Model

- **Architecture:** YOLOv8-Nano Segmentation (`yolov8n-seg`)
- **Framework:** [Ultralytics](https://github.com/ultralytics/ultralytics)
- **Task:** Instance segmentation (detection + pixel-level masks)

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Image size | 1250 px |
| Batch size | 8 |
| Epochs | 10+ (configurable) |
| Patience | 0 (no early stopping) |
| Base weights | `yolov8n-seg.pt` (pretrained) |

### Inference

Predictions are run with a configurable confidence threshold (default 0.2). Overlapping masks per class are merged via logical OR to produce clean segmentation outputs.

## Visualization Tools

- **`visuals/visualise_coco.py`** — Renders COCO annotations on images with bounding boxes, polygon masks, and category labels.
- **`visuals/display_yolo.py`** — Renders YOLO polygon annotations on images by denormalizing coordinates back to image dimensions.

## Project Structure

```
CSCI447_FinalProject/
├── README.md
├── requirements.txt
├── data/
│   ├── input/                  # Raw images and masks (train/val/test)
│   ├── coco/                   # COCO JSON annotations
│   └── yolo/                   # YOLO format labels + data.yaml
├── notebooks/
│   └── YOLOv8.ipynb            # Training and inference notebook
├── utils/
│   ├── train_val_split.py      # Dataset splitting
│   ├── masks_to_coco.py        # Mask → COCO conversion
│   └── coco_to_yolo.py         # COCO → YOLO conversion
└── visuals/
    ├── visualise_coco.py       # COCO annotation viewer
    └── display_yolo.py         # YOLO annotation viewer
```

## Getting Started

### Requirements

```bash
pip install -r requirements.txt
```

Dependencies: `ultralytics`, `opencv-python`

### Running the Pipeline

1. Place raw images and masks in `data/input/`.
2. Split the data:
   ```bash
   python utils/train_val_split.py
   ```
3. Convert masks to COCO, then to YOLO:
   ```bash
   python utils/masks_to_coco.py
   python utils/coco_to_yolo.py
   ```
4. Open `notebooks/YOLOv8.ipynb` and run training/inference cells.
