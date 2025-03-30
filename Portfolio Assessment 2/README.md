# Portfolio Assessment 2: Systematic Approach to Develop ML Model

**Course:** COS40007 – Artificial Intelligence for Engineering Projects  
**Author:** Alvin Phan (KyanS05) – 104480130  
**Week:** 4

---

## Learning Aim

This portfolio focuses on building a complete machine learning pipeline from raw motion sensor data collected from boning and slicing tasks in a meat processing environment. The goal is to classify these two activities using statistical and composite feature engineering, followed by training and evaluating multiple classifiers.

By the end of this portfolio task, you should:
- Understand how to transform raw time-series sensor data into feature vectors
- Compute statistical and composite features using NumPy and pandas
- Train and evaluate machine learning models using scikit-learn
- Apply hyperparameter tuning, feature selection, and dimensionality reduction
- Compare multiple classifiers and justify model choices based on results

---

## Tasks Completed

### Step 1: Data Collection
- Loaded `Boning.csv` and `Slicing.csv` from raw motion sensor data
- Extracted Frame, Neck (x,y,z), and Head (x,y,z) columns based on student number ending in 0
- Labeled boning as class 0 and slicing as class 1
- Combined and saved as `step1_combined.csv`

### Step 2: Composite Feature Creation
- Created 6 composite features per sensor group:
  - RMS of XY, YZ, ZX, XYZ
  - Roll and Pitch
- Added 12 new columns (6 for Neck, 6 for Head)
- Saved result as `step2_composite.csv`

### Step 3: Statistical Feature Computation
- Grouped dataset into 1-minute blocks (60 frames)
- For each of 18 motion columns, computed:
  - Mean, Std, Min, Max, AUC (sum of abs), Peak Count
- Final dataset: 108 features + class label per row
- Saved result as `step3_features.csv`

### Step 4: Model Training
- Trained baseline and tuned SVM models using:
  - 70/30 train-test split
  - 10-fold cross-validation
- Applied SelectKBest for top 10 features and PCA for 10 components
- Trained other models (SGD, RF, MLP) for performance comparison
- Saved outputs for each stage in structured CSVs

### Step 5: Model Selection
- Compared all SVM variants and concluded `svm_top10_features` was best
- Random Forest achieved perfect accuracy and was selected as overall best model

---

## How to Use the Code

### Requirements

```bash
pip install numpy pandas scikit-learn
```

### Files Included
- `Portfolio 2/Boning.csv`, `Slicing.csv` → Raw input files
- `step1_combined.csv` → Frame + selected sensor columns + class label
- `step2_composite.csv` → Added composite features
- `step3_features.csv` → 108 statistical features per 1-minute block
- `step4.1_baseline.csv` to `step4.5_other_models.csv` → Evaluation outputs
- `step4.4a_summary_svm.csv`, `step4.5_other_models.csv` → Final comparison tables

### Key Functions and Tools
- `np.mean`, `np.std`, `np.sum`, `np.min`, `np.max` → Feature computation
- Custom peak count using `np.diff()` and sign change logic
- `train_test_split()`, `SVC()`, `GridSearchCV()` → Model training & tuning
- `SelectKBest()`, `PCA()` → Feature reduction
- `RandomForestClassifier`, `SGDClassifier`, `MLPClassifier` → Model comparison

### Running the Code
Run each step in order from Step 1 to Step 5. Each step generates data needed for the next. Outputs are saved as CSVs in `Portfolio 2/`.

---

## Notes
- This task follows the same systematic process as Studio 3, but applies it to a new raw dataset
- Studio 3 helped frame the structure and methods for feature engineering and evaluation
- Composite features and 1-minute statistics are essential to transforming raw signals into usable ML inputs
- Resulting CSV files can be directly reused in the final model selection and submission steps

