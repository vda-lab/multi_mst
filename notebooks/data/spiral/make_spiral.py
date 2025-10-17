import numpy as np

# Spiral length
Nl = 500
l_min = 2
l_max = 25

# Depth
Nz = 50
z_min = 0
z_max = 4
z_mid = (z_min + z_max) / 2

# Spiralling params
s = 0.03
e = 2

# Noise
n = 0.0395

l = np.power(np.linspace(np.power(l_min, e), np.power(l_max, e), Nl), 1/e)
z = np.linspace(z_min, z_max, Nz)
L, Z = np.meshgrid(l, z)
LZ = np.vstack((L.ravel(), Z.ravel())).T

X = s * L**e * np.cos(L)
Y = s * L**e * np.sin(L)
nx = X + np.random.normal(loc=0, size=L.shape, scale=n * L)
ny = Y + np.random.normal(loc=0, size=L.shape, scale=n * L)
nz = Z + np.random.normal(loc=0, size=L.shape, scale=n * L)
D = np.vstack((nx.ravel(), ny.ravel(), nz.ravel())).T

np.save('./data/spiral/generated/X_uniform.npy', D)
np.save('./data/spiral/generated/lz_uniform.npy', LZ)

l1 = 16.2
l2 = 17.895
l3 = 18.105
l4 = 19.8

mask = (
    ((LZ[:, 0] >= l2) & (LZ[:, 0] <= l3)) | 
    ((LZ[:, 0] > l1) & (LZ[:, 0] < l2) & ((np.abs(LZ[:, 1] - z_mid) / z_mid) > ((l2 - LZ[:, 0]) / (l2 - l1)))) |
    ((LZ[:, 0] > l3) & (LZ[:, 0] < l4) & ((np.abs(LZ[:, 1] - z_mid) / z_mid) > ((LZ[:, 0] - l3) / (l4 - l3))))
)
LZ = LZ[~mask]
D = D[~mask]

np.save('./data/spiral/generated/X.npy', D)
np.save('./data/spiral/generated/lz.npy', LZ)