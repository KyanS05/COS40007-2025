# Portfolio Assessment 4: AI-Based Wooden Log Detection Pipeline
**Course:** COS40007 – Artificial Intelligence for Engineering Projects  
**Author:** Alvin Phan (KyanS05) – 104480130  
**Assessment:** Portfolio 4

---

## 🚨 **IMPORTANT**: Access Full Code, Model Outputs & Dataset via Google Drive
**🔗 [Click here to open the Portfolio 4 folder](https://drive.google.com/drive/folders/1TtwhRUhblLzDKuNTeDRksYTKHiy_DETW?usp=sharing)**  
All scripts, models (except large `.pth` files), test images, and results are hosted here due to GitHub's 100MB limit.

---

## Learning Aim
This portfolio task focuses on the creation of a complete deep learning pipeline for object detection, adapted from Studio 5. Tasks include dataset preparation, CNN and ResNet50 classification, Mask R-CNN model training, prediction visualization, log counting, and multi-class relabeling using Labelme.

By the end of this portfolio:
- Logs should be detected using bounding boxes and masks
- Images should be classified and counted
- The pipeline should support new class extension (`detected_log`)

---

## Tasks Completed

### Task 1: CNN and ResNet50 Classification
- Script: `1.1_sorting.py`, `1.2_resnet50.py`
- Created train/test split with 10 test images
- Trained simple CNN and ResNet50 model
- Visualised predictions and exported results
- Output:
  - `resnet50_test_results.csv`
  - `cnn_test/`
  - `resnet50_test/`

### Task 2: Mask R-CNN Log Detection
- Scripts: `2.2_train_mask_rcnn.py`, `2.4_test_visualise.py`, `2.5_log_counts.py`
- Converted Labelme JSONs to COCO with `labelme2coco`
- Trained Mask R-CNN using COCO format annotations
- Visualised output (bounding boxes + confidence)
- Counted logs detected per image
- Output:
  - `rcnn_test/model_final.pth`
  - `rcnn_test/visualised_outputs/`
  - `rcnn_test/metrics.json`
  - `2.5_log_counts.py`

### Task 3: Multi-Class Relabeling with `detected_log`
- Folder: `task3_labelled/`
- Duplicated the 10 test images used in Task 2
- Manually updated broken log labels to `detected_log` in Labelme
- Prepared the dataset for future multi-class training

---

## Folder Structure
```
Portfolio 4/
├── code/                        # Python scripts by task
├── cnn_test/                    # CNN output images
├── resnet50_test/               # ResNet50 visualised outputs
├── resnet50_test_results.csv    # ResNet50 prediction log
├── rcnn_test/                   # Trained model, outputs, visualised detections
├── log_data/                    # Train/test images + COCO jsons
├── task3_labelled/              # Updated images with 'detected_log' class
├── data/                        # Original image split
```

---

## How to Run the Code
Ensure all files are inside `Portfolio 4/`. Run the following:
```bash
python code/1.1_sorting.py
python code/1.2_resnet50.py
python code/2.2_train_mask_rcnn.py
python code/2.4_test_visualise.py
python code/2.5_log_counts.py
```

---

## Key Outputs
- `model_final.pth` – final trained R-CNN model
- `metrics.json` – loss log from training
- `visualised_outputs/` – images with bounding boxes and masks
- `step4_rules.txt` – log count output from prediction script
- Labelled JSONs with multi-class output (`log`, `detected_log`)

---

## Notes
- All models run on CPU (macOS compatible)
- Dataset labelled with Labelme, converted via `labelme2coco`
- Output structured for submission and GitHub upload

---

**GitHub Repo:** [https://github.com/KyanS05/COS40007-2025/tree/main/Portfolio%20Assessment%204](https://github.com/KyanS05/COS40007-2025/tree/main/Portfolio%20Assessment%204)
