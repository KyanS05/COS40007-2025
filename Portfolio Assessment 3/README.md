# Week 4 Portfolio Assessment 3
**Course:** COS40007 – Artificial Intelligence for Engineering Projects  
**Author:** Alvin Phan (KyanS05) – 104480130  
**Assessment:** Portfolio 3  

---

## Learning Aim
This portfolio task develops and tests a complete ML pipeline based on lessons from Studio 4. Tasks include sampling, feature checking, composite creation, model training and comparison, AI deployment, and rule extraction via decision trees.

By the end of this task, the model should:
- Be trained on carefully prepared features
- Generalize to new unseen user data
- Be deployed like an AI system
- Be explainable using decision rules from SP features

---

## Tasks Completed

### Step 1: Data Preparation
- File: `step1.py`
- Loaded and shuffled `vegemite.csv`
- Extracted 1000 balanced rows for AI test
- Removed constant-value columns
- Detected columns with few unique values (possible categoricals)
- Output:
  - `step1_test_data.csv`
  - `step1_train_data.csv`
  - `step1_train_cleaned.csv`

### Step 2: Feature Selection + Model Evaluation
- File: `step2.py`
- Selected top 20 features using `mutual_info_classif`
- Trained 5 models: SVM, SGD, Random Forest, MLP, Decision Tree
- Compared using classification reports + confusion matrices
- Saved the best model: Random Forest
- Saved top 20 features for future reuse
- Output:
  - `best_model.pkl`
  - `top20_features.pkl`

### Step 3: AI from ML Deployment
- File: `step3.py`
- Loaded saved model and 1000 unseen test rows
- Applied top 20 features from Step 2
- Generated predictions + report on real user data
- Output:
  - Classification report
  - Confusion matrix plot

### Step 4: Rule Extraction (SP Features Only)
- File: `step4.py`
- Filtered SP-only columns from training data
- Trained DecisionTreeClassifier with `max_depth=5`
- Visualized tree + saved rule text output
- Output:
  - Tree plot (PNG)
  - `step4_rules.txt`

---

## How to Run the Code
Ensure files are in `Portfolio 3/` directory. Run scripts in order:
```bash
python step1.py
python step2.py
python step3.py
python step4.py
```

---

## Output Files
- step1_test_data.csv
- step1_train_data.csv
- step1_train_cleaned.csv
- best_model.pkl
- top20_features.pkl
- step4_rules.txt
- Confusion matrix screenshots from step2 and step3

---

## Notes
- All models use scikit-learn and matplotlib only
- Feature selection, saving, and reloading done using `pickle`
- Tree explainability only uses SP sensors for transparency

---

**GitHub Repo:** [Link to repo here]

