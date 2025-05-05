import os
import random
import shutil
import torch
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ✅ Added for IoU comparison
from shapely.geometry import box

def get_max_iou(gt_boxes, pred_boxes):
    max_iou = 0
    for gt in gt_boxes:
        for pred in pred_boxes:
            g = box(*gt)
            p = box(*pred)
            iou = g.intersection(p).area / g.union(p).area
            max_iou = max(max_iou, iou)
    return max_iou

# Import YOLOv5 classic training
import sys
sys.path.append('G:/AI Proj/yolov5-master')
from train import run as train_yolo
from detect import run as detect_yolo

# CONFIG 
root = 'G:/AI Proj'
dataset_dir = f'{root}/dataset'
yaml_path = f'{root}/data.yaml'
output_dir = f'{root}/output'
run_name = 'graffiti_final'

original_test_img = f'{root}/original/images/test'
original_test_lbl = f'{root}/labels/test'
test_img_dst = f'{dataset_dir}/images/test'
test_lbl_dst = f'{dataset_dir}/labels/test'

# 1: Prepare 40 random test images + labels
def prepare_test_data():
    if os.path.exists(test_img_dst):
        shutil.rmtree(test_img_dst)
    if os.path.exists(test_lbl_dst):
        shutil.rmtree(test_lbl_dst)
    os.makedirs(test_img_dst, exist_ok=True)
    os.makedirs(test_lbl_dst, exist_ok=True)

    all_imgs = [f for f in os.listdir(original_test_img) if f.lower().endswith(('.jpg', '.png'))]
    selected = random.sample(all_imgs, 40)

    for file in selected:
        base = os.path.splitext(file)[0]
        shutil.copy(os.path.join(original_test_img, file), os.path.join(test_img_dst, file))
        label_path = os.path.join(original_test_lbl, base + '.txt')
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(test_lbl_dst, base + '.txt'))

    return selected

# 2: Write data.yaml
def write_yaml():
    with open(yaml_path, 'w') as f:
        f.write(f"""train: {dataset_dir}/images/train
val: {dataset_dir}/images/test
nc: 1
names: ['graffiti']
""")

# 3: Train YOLOv5
def train_yolov5():
    train_dir = os.path.join(output_dir, run_name)
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)

    train_yolo(
        data=yaml_path,
        imgsz=640,
        batch=8,
        epochs=50,
        weights='yolov5s.pt',
        device=0,
        project=output_dir,
        name=run_name
    )

# 4: Run detection on 40 test images
def run_detection():
    pred_dir = os.path.join(output_dir, f'{run_name}_pred')
    if os.path.exists(pred_dir):
        shutil.rmtree(pred_dir)

    detect_yolo(
        weights=f'{output_dir}/{run_name}/weights/best.pt',
        source=test_img_dst,
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        save_txt=True,
        save_conf=True,
        project=output_dir,
        name=f'{run_name}_pred'
    )

# 5: Compute IoU using predictions
def compute_iou(image_names):
    pred_dir = f'{output_dir}/{run_name}_pred/labels'
    csv_rows = []

    for img_name in tqdm(image_names, desc="Evaluating"):
        img_base = Path(img_name).stem
        img_path = os.path.join(test_img_dst, img_name)
        pred_file = os.path.join(pred_dir, img_base + '.txt')
        gt_file = os.path.join(test_lbl_dst, img_base + '.txt')

        if not os.path.exists(gt_file):
            continue

        gt_boxes = []
        with open(gt_file, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                _, xc, yc, w, h = parts
                img = cv2.imread(img_path)
                ih, iw = img.shape[:2]
                x1 = (xc - w/2) * iw
                y1 = (yc - h/2) * ih
                x2 = (xc + w/2) * iw
                y2 = (yc + h/2) * ih
                gt_boxes.append([x1, y1, x2, y2])

        pred_boxes = []
        confs = []
        if os.path.exists(pred_file):
            with open(pred_file, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    _, xc, yc, w, h, conf = parts
                    img = cv2.imread(img_path)
                    ih, iw = img.shape[:2]
                    x1 = (xc - w/2) * iw
                    y1 = (yc - h/2) * ih
                    x2 = (xc + w/2) * iw
                    y2 = (yc + h/2) * ih
                    pred_boxes.append([x1, y1, x2, y2])
                    confs.append(conf)

        iou = get_max_iou(gt_boxes, pred_boxes) if pred_boxes and gt_boxes else 0
        conf_val = confs[0] if confs else 0
        csv_rows.append([img_name, conf_val, iou])

    df = pd.DataFrame(csv_rows, columns=['image_name', 'confidence', 'IoU'])
    df.to_csv(f'{output_dir}/{run_name}/iou_results.csv', index=False)
    print(f"✅ Saved IoU results to: {output_dir}/{run_name}/iou_results.csv")

    high_iou_count = (df['IoU'] > 0.9).sum()
    mean_iou = df['IoU'].mean()
    print(f"\n🎯 Summary for '{run_name}'")
    print(f"📈 IoU > 0.9: {high_iou_count}/40 images")
    print(f"📊 Mean IoU: {mean_iou:.4f}")

    if high_iou_count >= 32:
        print("✅ This model PASSED the 80% IoU benchmark 🎉")
    else:
        print("❌ This model did NOT pass the 80% benchmark. Try training again.")

# MAIN RUN
if __name__ == '__main__':
    print("📦 Copying 40 test images and labels...")
    selected = prepare_test_data()

    print("📝 Writing YAML config...")
    write_yaml()

    print("🚀 Training YOLOv5 model...")
    train_yolov5()

    print("📸 Running detection on test set...")
    run_detection()

    print("📊 Computing IoU for test set...")
    compute_iou(selected)

    print("✅ All done.")
