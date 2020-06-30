import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# LDA is also a dimensionality reduction method
# it maximises the variance among classes but at the same
# time, it minimizes the variance in between class
# in short, we want to maximise S_b/S_w

# to do that, we find the top eigen values and top eigen vectors
# to find the optimized weigt matrix
n_components = 2
class_1 = np.array([[2,3,2], [1,1,2], [1.5,2.5, 3.5],
                    [1.5, 2.5, 2.5], [2,2,2], [2.5,2.5,2.5],
                    [1.5,2,3],[1,2.5,1],[1,1,1]])

class_2 = np.array([[4,5,4],[4,5,5],[4,4,6],
                    [4.5,5.5,4],[5.5,5.5,5],[6,4,6],
                    [5,4,4],[4,5,6],[5,6,5]])

# First thing first, standardise the dataset
x = np.concatenate((class_1, class_2))

# d_std = (d-mean)/std
# standardise the dataset column-wise
x = (x - np.mean(x, axis = 0))/np.std(x, axis = 0)

# get standardised class 1 and class 2
class_1 = x[:9]
class_2 = x[9:]
# print(x)

# now find the covariance matrix within class
# basically it is just element-wise sum of the covariance matrix of classes in the dataset
cov_1 = np.cov(class_1.transpose())
cov_2 = np.cov(class_2.transpose())

S_w = cov_1 + cov_2

# now compute the in-between classes scatter
# but we need to compute the mean vectors of all classes first and then compute the global
# mean of the overall dataset
mean_1 = np.mean(class_1, axis = 0)
mean_2 = np.mean(class_2, axis = 0)
global_mean = np.mean(x, axis = 0)

S_b = np.zeros((x.shape[1], x.shape[1]))

data = [class_1, class_2]
mean_vecs = [mean_1, mean_2]
for i in range(len(data)):
    n = data[i].shape[0]

    mean_vec = mean_vecs[i]

    S_b += n * np.dot(mean_vec - global_mean, (mean_vec - global_mean).transpose())

# now find the eigen values and eigen vectors of S_b/S_w
eig_vals, eig_vecs = np.linalg.eig(S_b.dot(np.linalg.inv(S_w)))

top_eig_vals = eig_vals.argsort()[::-1] # sort in descending order
top_eig_vecs = eig_vecs[top_eig_vals][:,:n_components]

# print(top_eig_vecs)
# now find the resulting dataset after lda
y = np.matmul(x, top_eig_vecs)

y_1 = y[:9]
y_2 = y[9:]

# Visualizing the result
fig, ax = plt.subplots(figsize=(10,10))

ax.set_facecolor("black")
ax.grid(color='green')
ax.scatter(y_1[:,0], y_1[:,1], color='blue')
ax.scatter(y_2[:,0], y_2[:,1], color='red')

plt.show()
