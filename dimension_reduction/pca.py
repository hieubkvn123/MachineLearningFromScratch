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
print(final)

# (Optional) Implement SVM on top of the principal components
# initializes the weight vector
w = np.array([1/x.shape[1],1/x.shape[1]], dtype=np.float32)
b = 0.5
x = final.copy()
y = np.array([1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1])

# optimize the weight vector
losses = list()
variances = list()
ITERATIONS = 10000 # max iterations for training
LR = 0.001

for i in range(1, ITERATIONS + 1):
    # loop thru the training dataset
    loss = 0
    variance = 0

    # loop through all input vectors
    for j in range(x.shape[0]):
        # compute the gradient
        # check if 1 - y<x,w> > 0 (if yes -> misclassification)

        if(1 - y[j] * (np.dot(w, x[j]) + b) <= 0):
            for i_ in range(w.shape[0]): # update the elements in the weight vector
                w[i_] = w[i_] - LR * (2 * 1/i * w[i_]) # 1/i is the lr decay rate

        else: # misclassification occurred
            for i_ in range(w.shape[0]):
                w[i_] = w[i_] - LR * (2*1/i*w[i_] - y[j] * x[j][i_])

            b = b - LR * y[j] # update intercept

        predictions = np.dot(x[j], w) + b
        variance += (y[j] - predictions)**2

        loss += 1/i * (np.sqrt(w.dot(w))) + max(0, 1 - y[j] * (np.dot(w, x[j]) + b))

    # if the previous loss is smaller than the current loss
    # the model is converged
    if(len(losses) >= 1):
        if(losses[len(losses) - 1] < loss):
            break

    print("[INFO] Epoch number " + str(i) + " | Loss = " + str(loss))
    losses.append(loss)
    variances.append(variance)

# now visualize the results
fig, ax = plt.subplots(1,2,figsize=(10,5))

ax[0].set_facecolor("black")
ax[0].grid(color="green")
ax[0].set_title("Implemented PCA model")
ax[0].set_xlabel("Feature 1")
ax[0].set_ylabel("Feature 2")

print(final)
ax[0].scatter(final[:9,0], final[:9,1], label="Class 1")
ax[0].scatter(final[9:,0], final[9:,1], label="Class 2")

# Visualizing the separating plane
x_ = np.linspace(-2, 2)
y = (-w[0]*x_ - b)/w[1]
sv_1 = (1-w[0]*x_ - b)/w[1]
sv_2 = (-1-w[0]*x_ - b)/w[1]

ax[0].plot(x_,y, color='orange', alpha=0.6, label='Separating plane')
ax[0].plot(x_,sv_1, color='blue', alpha=0.6, label='Support Vector 1', linestyle='--')
ax[0].plot(x_,sv_2, color='red', alpha=0.6, label='Support Vector 2', linestyle='--')
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
