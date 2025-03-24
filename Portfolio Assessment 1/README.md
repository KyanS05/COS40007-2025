# Portfolio Assessment 1 – Week 2 ML Pipeline

**Course:** COS40007 – Artificial Intelligence for Engineering  
**Author:** Alvin Phan (KyanS05) – 104480130  
**Week:** 2  
**Project Title:** Hello Machine Learning for Engineering  

---

## How to Run the Project

This project implements a complete machine learning pipeline using the Combined Cycle Power Plant dataset. It includes class labelling, normalisation, feature engineering, feature selection, and decision tree model comparison.

### Order of Execution
Run the following scripts in sequence:

1. `data_analyse.py`
   - Read and construct graphs to shows correllations betweens feature data

2. `class_label.py`  
   - Converts the continuous `PE` variable into 4 class labels using quantile binning
   - Saves: `converted_powerplant.csv`

3. `normalise.py`  
   - Applies MinMax scaling to original features (`AT`, `V`, `AP`, `RH`)
   - Saves: `normalised_powerplant.csv`

4. `feature.py`  
   - Engineers new features (`AT_RH`, `V_AP`, `AT_squared`, `RH_inverse`)
   - Saves: `features_powerplant.csv`

5. `slct_feature.py`  
   - Selects top 3 relevant features (`AT`, `V`, `RH`) from engineered dataset
   - Saves: `selected_feature_powerplant.csv`

6. `slct_converted.py`  
   - Selects the same 3 features from raw dataset
   - Saves: `selected_converted_powerplant.csv`

7. `decision_tree.py`  
   - Trains decision tree models on 5 different datasets
   - Outputs classification accuracy for each model
   - Saves: `model_accuracy_results.csv`

---

## Scripts Overview

| Script Name            | Purpose                                  |
|------------------------|------------------------------------------|
| `class_label.py`       | Create class labels from target variable |
| `normalise.py`         | Normalize all original numeric features  |
| `feature.py`           | Engineer composite features              |
| `slct_feature.py`      | Select key features (normalized)         |
| `slct_converted.py`    | Select key features (raw values)         |
| `decision_tree.py`     | Train and compare decision tree models   |

---

## Output

- 5 prepared datasets in CSV format
- Final model accuracy comparison (`model_accuracy_results.csv`)
- Visualizations (bar plots, histograms) displayed during execution

---

## Notes

- All `.py` files assume that `Folds5x2_pp.csv` is located in the same working directory.
- Scripts should be executed in the order listed to maintain data flow.
- Results and visualizations are designed to be easily interpreted by the tutor reviewing this portfolio.
