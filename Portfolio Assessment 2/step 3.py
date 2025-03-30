import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# load data
df = pd.read_csv('Portfolio 2/step2_composite.csv')

# select columns 2 to 19 (skip frame and class)
feature_cols = df.columns[1:-1]

# storage
feature_rows = []

# process in 60-frame blocks
for i in range(0, len(df), 60):
    block = df.iloc[i:i+60]
    if len(block) < 60:
        continue

    features = []
    for col in feature_cols:
        values = block[col].values
        features.extend([
            np.mean(values),
            np.std(values),
            np.min(values),
            np.max(values),
            np.sum(np.abs(values)),      # auc approximation
            np.sum((np.diff(np.sign(np.diff(values))) < 0).astype(int))  # peak count
        ])

    # add class label from last row
    class_val = block['class'].values[-1]
    features.append(class_val)

    feature_rows.append(features)

# column names
stat_names = ['mean', 'std', 'min', 'max', 'auc', 'peaks']
final_cols = [f'{col}_{stat}' for col in feature_cols for stat in stat_names] + ['class']

# build final df
df_features = pd.DataFrame(feature_rows, columns=final_cols)
df_features.to_csv('Portfolio 2/step3_features.csv', index=False)
print("step 3 done: saved as step3_features.csv")
