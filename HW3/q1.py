import numpy as np
import scipy.sparse.linalg as sp_linalg
import matplotlib.pyplot as plt

# Set parameters
m = 2048
C = 1

# Generate random points
X = np.random.rand(m, 2)

# Compute the pairwise distances
distances = np.linalg.norm(X[:, np.newaxis] - X[np.newaxis, :], axis=2)

# Compute the weight matrix W
epsilon = C * (np.log(m)**(3/4)) / (m**(1/2))
W = np.where(distances <= epsilon, (np.pi * epsilon**2)**(-1), 0)

print(W)

# Compute the unnormalized graph Laplacian L
D = np.diag(np.sum(W, axis=1))
L = D - W

# Compute the first four eigenvectors of L
vals, vecs = sp_linalg.eigsh(L, k=4, which='SM')

# Visualize eigenvectors
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for i in range(4):
    ax = axes[i // 2, i % 2]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=vecs[:, i], cmap='viridis')
    ax.set_title(f'Eigenvector {i+1}')
    fig.colorbar(scatter, ax=ax)

# plt.show()
