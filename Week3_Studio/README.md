# Week 3 Studio Activities

**Course:** COS40007 – Artificial Intelligence for Engineering Projects  
**Author:** Alvin Phan (KyanS05) – 104480130  
**Week:** 3

---

## Learning Aim

This studio focuses on developing machine learning workflows, including model training, evaluation, hyperparameter tuning, feature selection, and dimensionality reduction. You will work with motion sensor data collected from meat plant workers to classify physical activities using different ML models.

By the end of this studio, you should:
- Combine and preprocess multiple datasets using pandas
- Train and evaluate SVM classifiers using scikit-learn
- Tune SVM hyperparameters using GridSearchCV
- Reduce features using SelectKBest and PCA
- Compare model performance using different classifiers and preprocessing strategies

---

## Tasks Completed

### Activity 1: Data Preparation
- Loaded four separate CSV files (`w1.csv` to `w4.csv`)
- Combined them using `pd.concat()` into `combined_data.csv`
- Shuffled the combined dataset using `sample(frac=1)` and saved as `all_data.csv`

### Activity 2: SVM Model (Default)
- Used SVM (`SVC()`) with default parameters
- Performed 70/30 train-test split using `train_test_split()`
- Measured accuracy with `accuracy_score()`
- Performed 10-fold cross-validation with `cross_val_score()`
- Saved results to `tut3.2_result.csv`

### Activity 3: Hyperparameter Tuning
- Used `GridSearchCV` to tune `C` and `gamma` for `rbf` kernel
- Ran 5-fold cross-validation internally within the grid search
- Used best parameters to retrain SVM and evaluate accuracy
- Saved results to `tut3.3_result.csv`

### Activity 4: Feature Selection
- Applied `SelectKBest(f_classif, k=100)` to keep top 100 features
- Reused best SVM parameters from Activity 3
- Evaluated using train-test split and 10-fold CV
- Saved results to `tut3.4_result.csv`

### Activity 5: Dimensionality Reduction (PCA)
- Applied `PCA(n_components=10)` to reduce feature space
- Trained and evaluated tuned SVM using PCA-transformed data
- Saved results to `tut3.5_result.csv`

### Activity 6: Summary Table
- Combined results from Activities 2–5 into a single table
- Compared SVM performance under different preprocessing strategies
- Saved final summary as `tut3_summary_result.csv`

### Activity 7: Other Classifiers
- Trained and evaluated:
  - `SGDClassifier()`
  - `RandomForestClassifier()`
  - `MLPClassifier()`
- Used same data as Activity 2 (`all_data.csv`)
- Compared with baseline SVM
- Saved results to `tut3.7_result.csv`

---

## How to Use the Code

### Requirements

```bash
pip install pandas scikit-learn
```

### Files Included
- `Tutorial 3/w1.csv` to `w4.csv` → Raw datasets
- `Tutorial 3/combined_data.csv` → Concatenated file
- `Tutorial 3/all_data.csv` → Shuffled dataset for modeling
- `tut3.2_result.csv` → Base SVM
- `tut3.3_result.csv` → Tuned SVM
- `tut3.4_result.csv` → Feature-selected SVM
- `tut3.5_result.csv` → PCA-based SVM
- `tut3_summary_result.csv` → Comparison table for SVM models
- `tut3.7_result.csv` → Comparison of SVM, SGD, RF, and MLP

### Key Functions and Tools
- `train_test_split()` – create train/test partitions
- `SVC()` – support vector machine model
- `GridSearchCV()` – grid search for hyperparameter tuning
- `cross_val_score()` – k-fold model evaluation
- `SelectKBest()` – feature selection
- `PCA()` – dimensionality reduction
- `SGDClassifier()`, `RandomForestClassifier()`, `MLPClassifier()` – alternative classifiers

### Running the Code
Scripts should follow the activity sequence. Each block builds on the output of the previous one. Results are stored in CSV files for reuse in summary and comparison.

---

## Notes
- Each activity runs independently, but uses the same data source (`all_data.csv`)
- Activity 3 tuning may take a few minutes depending on system specs
- Feature selection and PCA help reduce overfitting and training time
- Final summary tables help compare model tradeoffs in accuracy and complexity
