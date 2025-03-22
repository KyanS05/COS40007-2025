# Week 1 Studio Activities

**Course:** COS40007 – Artificial Intelligence for Engineering Projects  
**Author:** Alvin Phan (KyanS05) – 104480130  
**Week:** 1

---

## Learning Aim
This week's studio focuses on developing foundational skills in Python for data analysis and exploratory data analysis (EDA). Students learn to load, clean, and explore a dataset, as well as understand the relationships between variables through univariate and multivariate analysis.

By the end of this week, you should:
- Understand the structure and purpose of the Combined Cycle Power Plant dataset
- Perform data cleaning (removing duplicates, checking for nulls)
- Conduct univariate and multivariate EDA using visualizations
- Interpret key statistical relationships using plots and correlation

---

## Tasks Completed

### Task 1: Initial Dataset Inspection
- Used `tut1.py`
- Loaded dataset `Folds5x2_pp.csv`
- Inspected the first five rows and dataset structure
- Checked for missing values and summary statistics

### Task 2: Data Cleaning
- Used `tut1.2.py`
- Removed 41 duplicate rows
- Verified that no missing values exist
- Cleaned and stripped whitespace from column names
- Validated dataset shape and column types

### Task 3: Exploratory Data Analysis (EDA)
- Used `tut1.3py.py`
- Plotted histograms for all features to assess distributions
- Created boxplots to detect potential outliers
- Generated correlation heatmap to assess relationships between features and target
- Created pairplot for deeper understanding of feature interactions

### Summary Report
- Generated EDA Summary Report (`Tutorial 1 EDA Report.pdf`)
- Found that Ambient Temperature (AT) has the strongest negative correlation with Power Output (PE)
- Identified outliers in features V (Vacuum) and RH (Humidity)
- Concluded the dataset is clean and ready for modeling

---

## How to Use the Code

### Requirements
Install the required libraries using:
```bash
pip install pandas matplotlib seaborn
```

### Files Included
- `tut1.py` → Initial data inspection
- `tut1.2.py` → Data cleaning and validation
- `tut1.3py.py` → EDA visualizations and correlation analysis
- `Folds5x2_pp.csv` → Raw dataset
- `Tutorial 1 EDA Report.pdf` → Final written summary of findings

### Running the Scripts
Run each script one at a time in the following order:
```bash
python tut1.py
python tut1.2.py
python tut1.3py.py
```

### Expected Outputs
- Console outputs showing dataset shape, info, and summary statistics
- Visual plots: histograms, boxplots, heatmap, and pairplot
- Written EDA summary in the form of a PDF report

---

## Notes
- All scripts assume `Folds5x2_pp.csv` is in the same directory
- Run scripts sequentially to reflect a logical data analysis pipeline
- Use the EDA report to guide future machine learning model development

