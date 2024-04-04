import numpy as np
from .noisy_mst import NoisyMST, noisyMST

# Force JIT compilation on import
random_state = np.random.RandomState(42)
random_data = random_state.random(size=(50, 3))
NoisyMST().fit(random_data)

__all__ = ["NoisyMST", "noisyMST"]
