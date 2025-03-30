import pandas as pd
import numpy as np

# load step 1 data
df = pd.read_csv('Portfolio 2/step1_combined.csv')

# function for RMS
def rms(*args):
    return np.sqrt(np.mean(np.square(args), axis=0))

# function for roll and pitch
def roll(y, x, z):
    return np.degrees(np.arctan2(y, np.sqrt(x**2 + z**2)))

def pitch(x, y, z):
    return np.degrees(np.arctan2(x, np.sqrt(y**2 + z**2)))

# compute composites for Neck
df['Neck_rms_xy'] = rms(df['Neck x'], df['Neck y'])
df['Neck_rms_yz'] = rms(df['Neck y'], df['Neck z'])
df['Neck_rms_zx'] = rms(df['Neck z'], df['Neck x'])
df['Neck_rms_xyz'] = rms(df['Neck x'], df['Neck y'], df['Neck z'])
df['Neck_roll'] = roll(df['Neck y'], df['Neck x'], df['Neck z'])
df['Neck_pitch'] = pitch(df['Neck x'], df['Neck y'], df['Neck z'])

# compute composites for Head
df['Head_rms_xy'] = rms(df['Head x'], df['Head y'])
df['Head_rms_yz'] = rms(df['Head y'], df['Head z'])
df['Head_rms_zx'] = rms(df['Head z'], df['Head x'])
df['Head_rms_xyz'] = rms(df['Head x'], df['Head y'], df['Head z'])
df['Head_roll'] = roll(df['Head y'], df['Head x'], df['Head z'])
df['Head_pitch'] = pitch(df['Head x'], df['Head y'], df['Head z'])

# save
df.to_csv('Portfolio 2/step2_composite.csv', index=False)
print("step 2 done: saved as step2_composite.csv")
