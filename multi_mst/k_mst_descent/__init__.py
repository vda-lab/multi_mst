import numpy as np
from .api import KMSTDescent, kMSTDescent

# Force JIT compilation on import
random_state = np.random.RandomState(42)
random_data = random_state.random(size=(50, 3))
KMSTDescent().fit(random_data)

__all__ = ["KMSTDescent", "kMSTDescent"]
