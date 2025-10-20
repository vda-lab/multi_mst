import numpy as np
import pandas as pd

# pandas read csv from URL:
url = "https://raw.githubusercontent.com/jadsoncastro/UnifiedView/refs/heads/master/data/cardiotocography.data"
df = pd.read_csv(url, sep=",", header=None)
df.to_parquet(
    "notebooks/data/cardiotocography/sources/cardiotocography.parquet", index=False
)
np.save("notebooks/data/cardiotocography/generated/X.npy", df.iloc[:, :-1].values)
np.save("notebooks/data/cardiotocography/generated/y.npy", df.iloc[:, -1].values)
