# Vehicle Detection Module (YOLOv8)

Real-time vehicle detection, tracking, and dashboard analytics powered by YOLOv8 Nano (`yolov8n.pt`) and OpenCV.

## Features

- **Multi-Class Vehicle Detection**: Detects cars, buses, trucks, motorcycles, and bicycles.
- **HUD & Live Dashboard**: Real-time vehicle counting, HUD metrics overlay, and per-class stats display.
- **Multiple Execution Modes**: Support for live GUI preview, headless video processing, single image detection, and webcam stream.
- **Interactive Controls**: Pause (`P`), snapshot/screenshot saving (`S`), and instant exit (`Q` / `Esc`).

---

## Directory Structure

```
vehicle_detection/
├── weights/
│   └── yolov8n.pt             # Pre-trained YOLOv8 Nano weights
├── video/
│   ├── Cars1.mp4              # Test sample video 1
│   └── Cars2.mp4              # Test sample video 2
├── output/
│   └── .gitkeep               # Directory for saving annotated outputs & screenshots
├── src/
│   └── vehicle_detection.py   # Core detection & dashboard script
├── requirements.txt           # Python dependencies
├── .gitignore                 # Exclusion rules for virtual envs and output binaries
└── README.md                  # Module documentation
```

---

## Installation & Setup

1. **Navigate to the module directory**:
   ```bash
   cd models/vehicle_detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

Run the detection script with various modes from `models/vehicle_detection`:

### 1. Live Mode with Preview Window (Default)
```bash
python src/vehicle_detection.py live
```

### 2. Process Custom Video
```bash
python src/vehicle_detection.py live video/Cars1.mp4
```

### 3. Save Processed Video (Headless / No GUI window)
```bash
python src/vehicle_detection.py video video/Cars1.mp4
```

### 4. Process Single Image
```bash
python src/vehicle_detection.py image path/to/image.jpg
```

### 5. Live Webcam Detection
```bash
python src/vehicle_detection.py webcam
```

### 6. Test Model Load
```bash
python src/vehicle_detection.py test
```

---

## Interactive Controls (in Live Mode)

| Key | Action |
|---|---|
| **P** | Pause / Resume video playback |
| **S** | Capture & save current frame screenshot to `output/` |
| **Q / Esc** | Exit live preview |
