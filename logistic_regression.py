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
ITERATIONS = 1000  # The max number of training iterations
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
# first visualizing the sigmoid line
x = np.linspace(-12, 12)
y = sigmoid(x)

# some configurations
ax[0].set_facecolor("black")
ax[0].grid(color='green')
ax[0].plot(x, y, color = 'orange', alpha = 0.4, label = 'Sigmoid Line')

h_theta_x_class1 = list()
h_theta_x_class2 = list()
p_class_1 = list()
p_class_2 = list()

for i in range(class_1.shape[0]):
    h_theta_x_1 = forward(w, class_1[i], b)
    p_1 = sigmoid(h_theta_x_1)

    h_theta_x_2 = forward(w, class_2[i], b)
    p_2 = sigmoid(h_theta_x_2)

    h_theta_x_class1.append(h_theta_x_1)
    p_class_1.append(p_1)
    h_theta_x_class2.append(h_theta_x_2)
    p_class_2.append(p_2)

ax[0].scatter(h_theta_x_class1, p_class_1, color='blue', label='Class 1')
ax[0].scatter(h_theta_x_class2, p_class_2, color='red', label='Class 2')
ax[0].hlines(0.5, -12, 12, color='cyan', linestyle='--')

# do some filling over here
x = np.linspace(-12, 0)
y_horizontal_line = 0 * x + 0.5
y_sigmoid_half = sigmoid(x)
ax[0].fill_between(x, y_horizontal_line, y_sigmoid_half, color='red', alpha=0.4)

x = np.linspace(0, 12)
y_horizontal_line = 0 * x + 0.5
y_sigmoid_half = sigmoid(x)
ax[0].fill_between(x, y_horizontal_line, y_sigmoid_half, color='blue', alpha = 0.4)

ax[0].set_title("Results Visualization")
ax[0].set_xlabel("Sigmoid of kernel")
ax[0].set_ylabel("Probability")
ax[0].legend()

ax[1].set_facecolor("black")
ax[1].grid(color='green')
ax[1].plot(losses, color='red', label='Losses')
ax[1].plot(variances, color='yellow', label='Variances')

ax[1].fill_between(list(range(len(losses))), losses, color='red', alpha=0.4)
ax[1].fill_between(list(range(len(variances))), variances, color='yellow', alpha=0.8)

ax[1].set_xlabel("Epochs/Iterations")
ax[1].set_ylabel("Variances/Losses")
ax[1].legend()

plt.show()
