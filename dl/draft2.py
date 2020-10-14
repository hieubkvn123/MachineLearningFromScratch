import numpy as np

lr = 1e-3
def sigmoid(x):
    exp = np.exp(-x)
    return 1 / (1 + exp)

def der(x, func=sigmoid):
    eps = 1e-8
    dy = func(x + eps) - func(x)
    dx = eps

    return dy/dx

inputs = np.array([[2,3], [1,2], [2,2], [1,0], [2,1]])
outputs = np.array([[1,0,0], [0,1,0], [1,0,0], [0,0,1],[0,1,0]])
weights = np.ones((2,3)) - 0.34

z = inputs @ weights
a = sigmoid(z)
predictions = sigmoid(inputs @ weights)

for i  in range(10000):
    z = inputs @ weights
    a = sigmoid(z)
    
    gradient = np.multiply(2*(predictions-outputs), der(z))
    gradient = inputs.transpose() @ gradient

    weights = weights - lr * gradient

    loss = (a - outputs) ** 2
    print('[*] Loss = %.4f' % loss.mean())
