import pandas as pd

# load data
w1 = pd.read_csv('Tutorial 3/w1.csv')
w2 = pd.read_csv('Tutorial 3/w2.csv')
w3 = pd.read_csv('Tutorial 3/w3.csv')
w4 = pd.read_csv('Tutorial 3/w4.csv')

# combine data
combined_data = pd.concat([w1, w2, w3, w4], ignore_index=True)
combined_data.to_csv('Tutorial 3/combined_data.csv', index=False)

# shuffle and save
shuffled_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)
shuffled_data.to_csv('Tutorial 3/all_data.csv', index=False)