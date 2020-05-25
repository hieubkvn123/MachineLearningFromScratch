import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

class_1 = np.array([[2,3], [1,2], [2.5, 3.5],
                    [1.5, 2.5], [2,2], [2.5,2.5],
                    [1.5,3],[2.5,1],[1,1]])

class_2 = np.array([[4,5],[5,5],[4,6],
                    [4.5,5.5],[5.5,5.5],[6,6],
                    [5,4],[5,6],[6,5]])

# sigmoid(x) = 1/(1 + e**(-x))
# initialize the weight vector
w = np.array([1,1], dtype=np.float32)
b = 1.0
x = np.concatenate((class_1, class_2))
y = np.array([1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])

# define a sigmoid function
def sigmoid(x):
    sigma = 1/(1 + np.exp(-x))
    return sigma

def forward(theta, x, b):
    dot_product = np.dot(theta, x)
    prediction = dot_product + b

    return prediction

def h_theta_x(theta, x, b):
    z = forward(theta, x, b)
    h_x_i = sigmoid(z)

    return h_x_i

LR = 0.01          # The learning rate
ITERATIONS = 10000 # The max number of training iterations
MIN_LOSS = 0.05    # The min loss to be considered converged

# The training process goes as followed :
# 1. Compute the loss and compare it to the previous iteration
#   --> If improved -> we proceed to update weights
#   --> If not improved -> The model is considered converged -> stop training

previous_loss = 0
variances = list()
losses = list()

for i in range(ITERATIONS):
    loss = 0
    variance = 0

    # L = -y.log(h_theta_x) - (1-y).log(1-h_theta_x) 
    # h_theta_x = sigmoid(<x,w>)

    for j in range(x.shape[0]):
        loss += -y[j] * np.log(h_theta_x(w, x[j], b)) - (1-y[j])*np.log(1-h_theta_x(w, x[j], b))
        prediction = h_theta_x(w, x[j], b)
        variance += (y[j] - prediction) ** 2

        # update the weight vector
        # by each coefficient
        for i_ in range(w.shape[0]):
            h_t_x = h_theta_x(w, x[j], b)
            w[i_] = w[i_] - LR*(h_t_x - y[j]) * x[j][i_]
            b = b - LR * (h_t_x - y[j])

    if( i != 0  and previous_loss < loss): # converged
        break
    else:
        if(loss <= MIN_LOSS):
            break

    previous_loss = loss
    losses.append(loss)
    variances.append(variance)

    print("[INFO] Epoch number " + str(i+1) + " | Loss = " + str(1/x.shape[0] * loss))


# Visualizing the results
fig, ax = plt.subplots(1, 2, figsize=(10,5))

# visualizing the results

