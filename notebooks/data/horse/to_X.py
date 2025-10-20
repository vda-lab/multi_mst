import pandas as pd
import numpy as np

data = pd.read_csv('./notebooks/data/horse/sources/horse.csv')
np.save('./notebooks/data/horse/generated/X.npy', data.to_numpy())