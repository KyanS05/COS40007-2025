# 2.1a.py - Graffiti Detection from Video
# Uses pretrained YOLOv5s model to detect graffiti in 4 raw videos

import torch
import cv2
import os

# Setup paths
video_dir = "G:/AI Proj/videos raw"
output_dir = "G:/AI Proj/video_output"
model_path = "G:/AI Proj/output/graffiti_final/weights/best.pt"  # path to trained model

os.makedirs(output_dir, exist_ok=True)

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
model.conf = 0.25

# List videos
videos = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

for video_file in videos:
    video_path = os.path.join(video_dir, video_file)
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(output_dir, f"boxed_{video_file}")
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        results = model(frame)
        boxes = results.xyxy[0]  # (x1, y1, x2, y2, conf, cls)

        for *xyxy, conf, cls in boxes:
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Processed {video_file} -> {output_path}")

print("✅ All videos processed.")
