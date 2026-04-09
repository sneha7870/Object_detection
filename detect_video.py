"""
Deep Learning Project — Video Object Detection using YOLOv8
Run: python detect_video.py
"""

import cv2
import torch
import numpy as np
import os
import time
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import defaultdict


VIDEO_PATH  = "27260-362770008_medium.mp4"
OUTPUT_PATH = "output/detected_output.mp4"
MODEL_NAME  = "yolov8n.pt"   # nano=fast | yolov8s/m/l/x.pt = more accurate
CONFIDENCE  = 0.40           # detection threshold (0.0 – 1.0)
IOU_THRESH  = 0.45           # NMS IoU threshold
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","TV","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8)


def draw_detections(frame, boxes, scores, class_ids):
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        color = COLORS[cls_id % len(COLORS)].tolist()
        label = f"{COCO_CLASSES[cls_id]}: {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame


def detect_video():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n📹 Video : {VIDEO_PATH}  ({width}x{height}, {fps} fps, {total} frames)")
    print(f"🖥️  Device: {DEVICE}\n")

    model = YOLO(MODEL_NAME)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    detection_log = defaultdict(int)
    fps_list      = []
    frame_count   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        results   = model.predict(source=frame, conf=CONFIDENCE,
                                  iou=IOU_THRESH, device=DEVICE, verbose=False)
        result    = results[0]
        boxes     = result.boxes.xyxy.cpu().numpy()
        scores    = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for cid in class_ids:
            detection_log[COCO_CLASSES[cid]] += 1

        annotated = draw_detections(frame.copy(), boxes, scores, class_ids)
        elapsed   = time.time() - t0
        fps_val   = 1.0 / elapsed if elapsed > 0 else 0
        fps_list.append(fps_val)

        cv2.putText(annotated,
                    f"FPS:{fps_val:.1f}  Frame:{frame_count+1}/{total}  "
                    f"Objects:{len(boxes)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        writer.write(annotated)
        frame_count += 1

        if frame_count % 20 == 0:
            print(f"  [{frame_count}/{total}]  fps={fps_val:.1f}  "
                  f"detections={len(boxes)}")

    cap.release()
    writer.release()
    print(f"\nAnnotated video saved → {OUTPUT_PATH}")
    return detection_log, fps_list, frame_count


def plot_stats(detection_log, fps_list, frame_count):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("YOLOv8 Object Detection — Statistics", fontsize=14, fontweight="bold")

    if detection_log:
        items = sorted(detection_log.items(), key=lambda x: x[1], reverse=True)[:15]
        labels, counts = zip(*items)
        axes[0].barh(labels, counts, color="steelblue")
        axes[0].set_xlabel("Total Detections")
        axes[0].set_title("Top Detected Objects")
        axes[0].invert_yaxis()

    axes[1].plot(fps_list, color="green", linewidth=1)
    axes[1].axhline(np.mean(fps_list), color="red", linestyle="--",
                    label=f"Avg FPS: {np.mean(fps_list):.1f}")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("FPS")
    axes[1].set_title("Inference Speed per Frame")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("output/detection_stats.png", dpi=150)
    plt.show()
    print("📊 Stats chart saved → output/detection_stats.png")


if __name__ == "__main__":
    print("=" * 55)
    print("   Deep Learning — Video Object Detection (YOLOv8)")
    print("=" * 55)

    detection_log, fps_list, frame_count = detect_video()

    print("\n── Detection Summary─")
    for obj, cnt in sorted(detection_log.items(), key=lambda x: -x[1]):
        print(f"   {obj:<22}: {cnt}")
    print(f"\n   Total frames   : {frame_count}")
    print(f"   Average FPS    : {np.mean(fps_list):.2f}")

    plot_stats(detection_log, fps_list, frame_count)
