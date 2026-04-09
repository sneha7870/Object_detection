# 🎯 Video Object Detection — YOLOv8
Deep Learning Assignment

## 📁 Files
```
object_detection/
├── detect_video.py                    ← Main script
├── requirements.txt                   ← Dependencies
├── 27260-362770008_medium.mp4         ← Your input video
└── output/
    ├── detected_output.mp4            ← Annotated output video (generated)
    └── detection_stats.png            ← Charts (generated)
```

## ⚙️ Setup & Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run detection
```bash
python detect_video.py
```

That's it! The script will:
1. Auto-download YOLOv8 weights (first run only, ~6 MB)
2. Process every frame of the video
3. Save annotated video → `output/detected_output.mp4`
4. Show + save detection statistics → `output/detection_stats.png`

## 🔧 Configuration (top of detect_video.py)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `CONFIDENCE` | 0.40 | Min detection confidence (lower = more detections) |
| `MODEL_NAME` | yolov8n.pt | Model size (n/s/m/l/x — bigger = more accurate) |
| `DEVICE` | auto | Uses GPU if available, else CPU |

## 🧠 How It Works
1. **YOLOv8** (You Only Look Once v8) processes each frame as a whole image
2. **Backbone (CSPDarknet)** extracts multi-scale feature maps
3. **Neck (PANet)** fuses features across scales
4. **Head** predicts bounding boxes + class probabilities simultaneously
5. **NMS** (Non-Maximum Suppression) removes duplicate detections
6. Results drawn on frame with class name + confidence score

## 📦 Model Options
| Model | Speed | Accuracy |
|-------|-------|----------|
| yolov8n.pt | ⚡ Fastest | Good |
| yolov8s.pt | Fast | Better |
| yolov8m.pt | Medium | Best for assignment |
| yolov8l.pt | Slow | High |
