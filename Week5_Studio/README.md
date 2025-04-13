# Week 5 Studio Activities

**Course:** COS40007 – Artificial Intelligence for Engineering Projects  
**Author:** Alvin Phan (KyanS05) – 104480130  
**Week:** 5

---

## Learning Aim

This studio introduces deep learning techniques with a focus on image classification and object detection using Convolutional Neural Networks (CNNs). It also covers the use of transfer learning with pretrained models (ResNet50) and hands-on image labelling using LabelMe for both classification and object-level detection tasks.

By the end of this studio, you should:
- Understand how CNNs are used for image classification
- Train a basic CNN model using the MNIST dataset
- Apply transfer learning using ResNet50 for a binary classification task
- Annotate image datasets using LabelMe at both image-level and object-level
- Visualise object detection annotations from polygon labels

---

## Tasks Completed

### Activity 1: Data Labelling
- **Rust Dataset**
  - Downloaded 10 unlabeled metal surface images
  - Manually sorted into `rust/` and `no_rust/` folders based on visible corrosion
  - Created a new folder structure with `train/` and `val/` subfolders

- **Log Dataset**
  - Used `labelme` to annotate wooden logs in 5 images
  - Saved each annotation as `.json` alongside its matching `.png` image in `logs/`

### Activity 2: Custom CNN with MNIST
- Loaded the MNIST dataset using `tensorflow.keras.datasets`
- Normalized grayscale images and reshaped them for CNN input
- Built a CNN with `Conv2D`, `MaxPooling2D`, and `Dense` layers
- Trained the model for 5 epochs with accuracy >99%
- Visualized training and validation accuracy using matplotlib

### Activity 3: Transfer Learning with ResNet50
- Loaded a pre-trained ResNet50 model (without top layer)
- Froze base layers and added custom dense layers for binary classification
- Trained on the rust/no_rust dataset using Keras `flow_from_directory()`
- Evaluated model performance (noted limitations due to small dataset)

### Activity 4: Object Detection Visualisation
- Parsed LabelMe `.json` files to extract polygon annotations
- Matched each `.json` with its corresponding `.png` image
- Drew annotated regions using `matplotlib.patches.Polygon`
- Verified successful object-level labelling for 5 images
- Also implemented a script to loop through and display all annotated images

---

## How to Use the Code

### Requirements

```bash
pip install tensorflow labelme opencv-python matplotlib numpy
```

### Files Included
- `rust_data/train/rust/`, `rust_data/train/no_rust/`  
- `rust_data/val/rust/`, `rust_data/val/no_rust/`  
- `logs/img_0001.png` to `img_0005.png` and matching `.json` files  
- `tut5.2.py` → MNIST model training
- `tut5.3.py` → Transfer learning with ResNet50
- `tut5.4.py` → Log annotation visualiser for LabelMe

### Key Functions and Tools
- `Conv2D()`, `MaxPooling2D()`, `Dense()` – layers for CNNs
- `ResNet50()` – pretrained CNN model from Keras applications
- `ImageDataGenerator()` – for directory-based image loading
- `labelme` + `json` – annotation tool and parser
- `matplotlib.patches.Polygon` – to draw labelled polygons
- `cv2.imread()` + `cv2.cvtColor()` – load and convert images

### Running the Code
- Run each `.py` script to execute training or visualisation per activity
- Ensure folder paths are consistent relative to script location
- Labelled logs must have matching `.png` and `.json` filenames

---

## Notes
- Due to small dataset size, model performance is not optimal (val_acc ~50%)
- Studio 5 demonstrates the full workflow from annotation to model training
- Object detection work can be extended using YOLO or Mask R-CNN for Portfolio 3
- LabelMe annotations are crucial for later model fine-tuning and deployment
