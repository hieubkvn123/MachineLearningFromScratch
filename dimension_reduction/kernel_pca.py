import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA

X, Y = make_circles(n_samples=200)
X_ = np.zeros((200, 3))
X_[:, :2] = X
X_[:, 2] = np.sqrt(X_[:, 0] ** 2 + X_[:, 1] ** 2)

fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)

for y in np.unique(Y):
    cluster = X_[Y == y]
    ax1.scatter3D(cluster[:, 0], cluster[:, 1], cluster[:, 2])

X_ = KernelPCA(n_components=2).fit_transform(X_)

for y in np.unique(Y):
    cluster = X_[Y == y]
    ax2.scatter(cluster[:, 0], cluster[:, 1])

plt.show()
