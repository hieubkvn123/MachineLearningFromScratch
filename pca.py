import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# PCA is a dimensionality reduction method
# that maximises the variance of the original dataset
# on a lower dimension projection

# to do that, we find the eigen values and vectors of the covariance matrix

# number of components to reduce to
n_components = 2
class_1 = np.array([[2,3,2], [1,1,2], [1.5,2.5, 3.5],
                    [1.5, 2.5, 2.5], [2,2,2], [2.5,2.5,2.5],
                    [1.5,2,3],[1,2.5,1],[1,1,1]])

class_2 = np.array([[4,5,4],[4,5,5],[4,4,6],
                    [4.5,5.5,4],[5.5,5.5,5],[6,4,6],
                    [5,4,4],[4,5,6],[5,6,5]])

# 1. Standardise the dataset
x = np.concatenate((class_1, class_2))

# d_std = (d - mean) / std
# we have to standardise the dataset column-wise
x = (x - np.mean(x, axis = 0))/np.std(x, axis=0)

# now put the whole dataset into a pandas dataframe
x = pd.DataFrame(data=x, columns = ['f1', 'f2','f3'])

# now find the covariance matrix of the dataframe
cov = np.cov(x.T)

# print(cov)
# find the eigen vectors and values of the covariance matrix
eig_vals, eig_vecs = np.linalg.eig(cov)

# sort the eigen values
eig_vals = eig_vals.argsort()[::-1]

# sort the eigen vectors according to the eigen values
eig_vecs = eig_vecs[eig_vals]

# now select the top 2 eigen vecs
top_eig_vecs = eig_vecs[:, :n_components]

# now find the principal components
final = np.dot(x, top_eig_vecs)

# (Optional) Implement SVM on top of the principal components


# now visualize the results
fig, ax = plt.subplots(1,2,figsize=(10,5))

ax[0].set_facecolor("black")
ax[0].grid(color="green")
ax[0].set_title("Implemented PCA model")
ax[0].set_xlabel("Feature 1")
ax[0].set_ylabel("Feature 2")

ax[0].scatter(final[:9,0], final[:9,1], label="Class 1")
ax[0].scatter(final[9:,0], final[9:,1], label="Class 2")
ax[0].legend()


# comparing with sklearn model
# the sign of the eigen vectors in the sklearn will be reversed
# but it is okay
pca = PCA(n_components = 2)
out = pca.fit_transform(x)

ax[1].set_facecolor("black")
ax[1].grid(color="green")
ax[1].set_title("PCA model from sklearn")
ax[1].set_xlabel("Feature 1")
ax[1].set_ylabel("Feature 2")

ax[1].scatter(out[:9, 0], out[:9,1], label='Class 1')
ax[1].scatter(out[9:, 0], out[9:,1], label='Class 2')
ax[1].legend()

plt.show()
