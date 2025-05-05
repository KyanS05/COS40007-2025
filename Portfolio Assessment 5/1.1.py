import pandas as pd
import os
import cv2

def convert(csv_path, image_dir, output_dir, class_id=0):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    grouped = df.groupby('filename')

    for filename, group in grouped:
        image_path = os.path.join(image_dir, filename)
        if not os.path.exists(image_path):
            print(f"Missing image: {image_path}")
            continue

        # Use OpenCV to read image size
        img = cv2.imread(image_path)
        if img is None:
            print(f"Unreadable image: {image_path}")
            continue
        img_height, img_width = img.shape[:2]

        yolo_lines = []
        for _, row in group.iterrows():
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            bbox_width = (xmax - xmin) / img_width
            bbox_height = (ymax - ymin) / img_height
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

        label_filename = os.path.splitext(filename)[0] + ".txt"
        with open(os.path.join(output_dir, label_filename), 'w') as f:
            f.write("\n".join(yolo_lines))

# === Convert both train and test sets ===
convert(
    csv_path='G:/AI Proj/Bounding_boxes/train_labels.csv',
    image_dir='G:/AI Proj/original/images/train/',
    output_dir='G:/AI Proj/labels/train/',
)

convert(
    csv_path='G:/AI Proj/Bounding_boxes/test_labels.csv',
    image_dir='G:/AI Proj/original/images/test/',
    output_dir='G:/AI Proj/labels/test/',
)
