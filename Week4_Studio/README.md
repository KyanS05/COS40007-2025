# Week 4 Studio Activities
**Course:** COS40007 – Artificial Intelligence for Engineering Projects  
**Author:** Alvin Phan (KyanS05) – 104480130  
**Week:** 4

---

## Learning Aim
This week's studio focuses on **model evaluation, class imbalance solutions, deployment simulation, and model interpretability**. Building on the dataset and process from Studio 3, this week introduces a more realistic view of using ML models as AI systems.

By the end of this studio, you should:
- Evaluate classifiers using precision, recall, F1-score, and confusion matrix
- Identify and fix class imbalance using SMOTE, TomekLinks, and class weighting
- Train and persist models using `pickle`
- Deploy a saved model on new unseen data (AI from ML)
- Explain model behavior using Decision Trees and mutual information

---

## Tasks Completed

### Activity 4.2: Classification Report
- Used `4.2.py`
- Trained and evaluated 4 models: SVM, SGD, RandomForest, MLP
- Compared performance using `classification_report`
- RandomForest achieved the best F1-score and accuracy overall
- Reports saved in `classification_reports.txt`

### Activity 4.3: Confusion Matrix
- Used `4.3.py`
- Plotted confusion matrices for all 4 models
- Visualized misclassifications, especially class 1 being predicted as class 2
- Confirmed need for class balancing

### Activity 4.4: Class Balancing
- Used `4.4.py`
- Applied SMOTE + TomekLinks to handle class imbalance
- Also tested manual class weighting `{0:0.3, 1:0.5, 2:0.2}`
- RandomForest retrained on balanced data reached 97% accuracy (class 1 recall = 1.0)
- Class weighting gave lower recall (~0.29), confirming SMOTE+Tomek was more effective

### Activity 4.5: Model Saving
- Used `4.5.py`
- Saved both SVM and RandomForest models trained on balanced data
- Saved as `mymodel1.pkl` (SVM) and `mymodel2.pkl` (RF)

### Activity 4.6: AI from ML (Real-Time Simulation)
- Used `4.6.py`
- Loaded the saved models
- Simulated row-by-row prediction on unseen data (`w4.csv`)
- Results:
  - SVM accuracy: 67%
  - RandomForest accuracy: 81%
- RandomForest showed stronger generalization to new users

### Activity 4.7: Model Explainability
- Used `4.7.py`
- Selected top 10 features using `mutual_info_classif`
- Trained and visualized a DecisionTreeClassifier
- Saved:
  - `decision_tree_model.pkl`
  - `decision_tree_text.txt` (text version of rules)
  - `feature_importance.csv` (ranked features by mutual info)

---

## How to Use the Code

### Requirements
Install the required libraries:
```bash
pip install pandas matplotlib scikit-learn imbalanced-learn
```

### Files Included
- `4.2.py` → Model training and classification report
- `4.3.py` → Confusion matrices
- `4.4.py` → Data balancing with SMOTE and class weights
- `4.5.py` → Model saving
- `4.6.py` → AI deployment simulation
- `4.7.py` → Model interpretability via Decision Tree

### Running the Scripts
Run each script sequentially from `4.2.py` to `4.7.py`:
```bash
python 4.2.py
python 4.3.py
python 4.4.py
python 4.5.py
python 4.6.py
python 4.7.py
```

### Expected Outputs
- Saved models (`mymodel1.pkl`, `mymodel2.pkl`, `decision_tree_model.pkl`)
- Confusion matrix visualizations
- Classification and decision rule text files
- Accuracy results on new unseen data

---

## Notes
- Ensure all four CSV files (`w1.csv`, `w2.csv`, `w3.csv`, `w4.csv`) are placed in the `Tutorial 4/` directory
- Run the scripts in order for consistency
- All output files are saved in the same directory and used by subsequent scripts

