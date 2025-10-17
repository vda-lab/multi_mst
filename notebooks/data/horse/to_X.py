import pandas as pd
import numpy as np

data = pd.read_csv('./notebooks/data/horse/source/horse.csv', header=None)
np.save('./notebooks/data/horse/generated/X.npy', data.values)