import numpy as np

x = np.array([2,1,3])

def f(x):
    x_ = x.copy()
    x_[0] *= 2
    x_[1] = x_[1] ** 3 + 2
    x_[2] = x_[2] ** 2 + x_[2] * 3

    return x_
def dy_dx(x):
    epsilon = 1e-8
    derivative = []

    for i in range(x.shape[0]):
        a_ = x + np.eye(x.shape[0])[i] * epsilon
        a  = x

        dy = f(a_)[i] - f(a)[i]
        dx = epsilon

        derivative.append(dy/dx)

    return np.array(derivative)

print(f(x), dy_dx(x))
