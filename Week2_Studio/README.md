# Week 2 Studio Activities
**Course:** COS40007 – Artificial Intelligence for Engineering Projects  
**Author:** Alvin Phan (KyanS05) – 104480130  
**Week:** 2

---

## Learning Aim
This week's studio focuses on **data pre-processing and feature engineering** for machine learning. The tasks follow a structured pipeline from labelling to model evaluation. Students will implement each stage of the pipeline manually to deepen their understanding of data transformation, modelling, and performance evaluation.

By the end of this week, you should:
- Understand and apply ground truth labelling
- Normalize and simplify feature values
- Engineer new features using statistical techniques
- Select relevant features based on correlation
- Train and evaluate a Decision Tree Classifier
- Compare and explain performance across multiple feature sets

---

## Tasks Completed

### Activity 2.1: Class Labelling / Creating Ground Truth
- Used `tut2.1.py`
- Converted numerical `strength` values into five categorical classes (1–5)
- Dropped the original `strength` column
- Saved as `converted_concrete.csv`
- Plotted strength class distribution using a bar chart

### Activity 2.2: Feature Engineering
- Used `tut2.2.py`
- Simplified the `age` feature into categories
- Normalized 7 numerical features using `MinMaxScaler`
- Created four composite features using pairwise combinations (multiplication or summation)
- Saved output as `features_concrete.csv` and `normalised_concrete.csv`

### Activity 2.3: Feature Selection
- Used `tut2.3.py`
- Selected top correlated features: Cement, Water, Superplastic, Age
- Saved normalized subset as `selected_feature_concrete.csv`

### Activity 2.4: Model Development & Evaluation
- Used `tut2.4.py`
- Created `selected_converted_concrete.csv` (unnormalized, selected features)
- Trained Decision Tree models using 5 datasets:
  1. `converted_concrete.csv`
  2. `normalised_concrete.csv`
  3. `features_concrete.csv`
  4. `selected_feature_concrete.csv`
  5. `selected_converted_concrete.csv`
- Logged model performance into `model_accuracy_results.csv`

### Activity 2.5: Summarisation
- Converted raw accuracy to percentages for easier interpretation
- Identified:
  - **Best Models:** Model 1 & 2 (99.68%)
  - **Weakest Model:** Model 5 (62.78%)
- Explained performance variation based on feature quality and preprocessing

---

## How to Use the Code

### Requirements
Ensure you have Python and required libraries:
```bash
pip install pandas matplotlib scikit-learn
```

### Files Included
- `tut2.1.py` → Ground truth labelling
- `tut2.2.py` → Normalisation + feature engineering
- `tut2.3.py` → Feature selection
- `tut2.4.py` → Model training & accuracy comparison
- `model_accuracy_results.csv` → Accuracy report for all datasets

### Running the Scripts
Run each script in sequence:
```bash
python tut2.1.py
python tut2.2.py
python tut2.3.py
python tut2.4.py
```

### Expected Outputs
- Transformed CSVs at each stage (`converted_`, `normalised_`, `features_`, etc.)
- Accuracy results per dataset
- Model performance summary

---

## Notes
- Ensure `concrete.csv` is placed in the `Tutorial 2/` directory before running any script
- Run the scripts in order (2.1 → 2.4)
- All output files are saved automatically to the same directory


