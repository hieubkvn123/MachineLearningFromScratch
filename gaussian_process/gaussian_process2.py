import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')

# Test data
n = 150
num_func = 20
random_x_y = True#False
Xtest = np.linspace(-5, 5, n).reshape(-1,1)

# Noiseless training data
Xtrain = np.array([-4, -3, -2, -1, 1, 4]).reshape(6,1)
ytrain = np.cos(np.sin((np.random.normal(loc=1, scale=5, size=(6,1)) * Xtrain)))
if(random_x_y):
    Xtrain = np.random.uniform(-5,5, size=(10,)).reshape(10,1)
    ytrain = np.random.uniform(0,10, size=(10,))

# Define the kernel function
def kernel(a, b, param):
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/param) * sqdist)

param = 0.3

K_11 = kernel(Xtrain, Xtrain, param=param)
K_12 = kernel(Xtrain, Xtest, param=param)
K_22 = kernel(Xtest, Xtest, param=param)

solved = np.linalg.solve(K_11, K_12)
mu = (solved.T @ ytrain).reshape(-1)
std = K_22 - (solved.T @ K_12)

y2 = np.random.multivariate_normal(mean=mu, cov=std, size=num_func)

for i in range(num_func):
    plt.plot(Xtest, y2[i, :], linewidth=0.5)
plt.plot(Xtest, mu, linestyle='--', linewidth=2, color='black')
plt.scatter(Xtrain, ytrain, s=90, color='red', marker='v')
plt.show()
