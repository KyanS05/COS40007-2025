import pandas as pd
from sklearn.utils import shuffle

# load data
df = pd.read_csv("Portfolio 3/vegemite.csv")
df = shuffle(df, random_state=42).reset_index(drop=True)

# check class counts
print("original class counts:")
print(df['Class'].value_counts())

# take 1000 balanced test samples
test_df = df.groupby('Class', group_keys=False).apply(lambda x: x.sample(n=333, random_state=42)).reset_index(drop=True)
train_df = df.drop(test_df.index).reset_index(drop=True)

# save splits
test_df.to_csv("Portfolio 3/step1_test_data.csv", index=False)
train_df.to_csv("Portfolio 3/step1_train_data.csv", index=False)

# find constant columns
const_cols = train_df.columns[train_df.nunique() == 1].tolist()
print("constant cols:", const_cols)
train_df = train_df.drop(columns=const_cols)

# find possible categorical cols
cat_cols = [col for col in train_df.columns if train_df[col].nunique() <= 10 and col != 'Class']
print("possible categorical cols:", cat_cols)

# check class balance
print("train class counts:")
print(train_df['Class'].value_counts())

# save cleaned train
train_df.to_csv("Portfolio 3/step1_train_cleaned.csv", index=False)
