# this is SVM implementation with
# the aid of numpy. Will make a version without numpy
# so that you people understand more about linear algebra

import numpy as np
import pandas as pd

# For those using matplotlib
import matplotlib
import matplotlib.pyplot as plt

# For those using seaborn
# import seaborn as sns

from sklearn.svm import SVC

# for linearly seperable cases, generate two classes of data
# that are very far away
class_1 = np.array([[2,3], [1,2], [2.5, 3.5],
                    [1.5, 2.5], [2,2], [2.5,2.5],
                    [1.5,3],[2.5,1],[1,1]])

class_2 = np.array([[4,5],[5,5],[4,6],
                    [4.5,5.5],[5.5,5.5],[6,6],
                    [5,4],[5,6],[6,5]])

# initialize the weight vector
w = np.array([1,1], dtype=np.float32)
b = 5.0
x = np.concatenate((class_1, class_2))
y = np.array([1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1])

# now optimize the weight vector
losses = list()
variances = list()
ITERATIONS = 100000 # max iterations of optimization
LR = 0.000001 # learning rate
LAMBDA = 0.001 # regularization parameter

for i in range(1, ITERATIONS + 1):
    # Loop thru the training dataset
    loss = 0
    variance = 0

    for j in range(x.shape[0]):
        # compute the gradient
        # check if 1 - y_i(w_t*x + b) > 0 (misclassification)
        # when there is no classification
        if(1 - y[j]*(np.dot(w, x[j]) + b) <= 0):
            for i_ in range(w.shape[0]): # update each coefficient
                w[i_] = w[i_] - LR * (2 * 1/i * LAMBDA * w[i_]) # 1/i is the lr decay rate

        else : # misclassification
            for i_ in range(w.shape[0]):
                w[i_] = w[i_] - LR * (2*1/i*LAMBDA*w[i_] - y[j] * x[j][i_])
                # b = b + LR * y[j] # update intercept

        predictions = np.dot(x[j], w) + b
        variance += (y[j] - predictions)**2

        loss += 1/i * (np.sqrt(w.dot(w))) + max(0, 1 - y[j] * (np.dot(x[j],w)+b))

    # if the previous loss is smaller than the 
    # current loss, the model is converged
    if(len(losses) >= 1):
        if(losses[len(losses) - 1] < loss):
            break

    print("[INFO] Epochs number " + str(i) + "| Loss = " + str(loss))
    losses.append(loss)
    variances.append(variance)

# visualize the results
fig, ax = plt.subplots(1,2,figsize=(10,5))

ax[0].set_facecolor("black")
ax[0].grid(color='green')
ax[0].scatter(class_1[:,0], class_1[:,1], color='blue', label = 'Class 2')
ax[0].scatter(class_2[:,0], class_2[:,1], color='red', label='Class 1')

# visualise the plane seperating the two classes
x = np.linspace(0, 8)
y = (-w[0] * x - b) * 1/(w[1])
ax[0].plot(x, y, color='orange')

# plot the support vector
support_vector_1 = (1 - w[0] * x - b) * 1/(w[1])
support_vector_2 = (-1 - w[0] * x - b) * 1/(w[1])
ax[0].plot(x, support_vector_1, color = 'blue', linestyle='--', label='Support Vector 1')
ax[0].plot(x, support_vector_2, color = 'red', linestyle='--', label='Support Vector 2')
ax[0].fill_between(x, y, support_vector_1, color='blue', alpha = 0.4)
ax[0].fill_between(x, y, support_vector_2, color='red', alpha = 0.4)
ax[0].legend()
ax[0].set_title("Results Visualization")
ax[0].set_xlabel("Feature 1")
ax[0].set_ylabel("Feature 2")

ax[1].set_facecolor("black")
ax[1].grid(color='green')
ax[1].plot(losses, color='red', label = 'Loss')
ax[1].plot(variances, color='yellow', label='Variance')
ax[1].fill_between(list(range(len(losses))), losses, color='red', alpha=0.8)
ax[1].fill_between(list(range(len(variances))), variances, color='yellow', alpha=0.4)
ax[1].legend()
ax[1].set_title("Loss/Variance Visualization")
ax[1].set_xlabel("Epochs/Iterations")
ax[1].set_ylabel("Loss/Variance")

print("-----------------------------------------------------------------")
print("[INFO] PLotting results ... ")
plt.show(block=True)
