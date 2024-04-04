import numpy as np
from .k_mst import KMST, kMST

# Force JIT compilation on import
random_state = np.random.RandomState(42)
random_data = random_state.random(size=(50, 3))
KMST().fit(random_data)

__all__ = ["KMST", "kMST"]
