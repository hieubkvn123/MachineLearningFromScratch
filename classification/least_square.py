import numpy as np
import matplotlib.pyplot as plt

'''
    In this example we will find the unique solution to the Least Square problem
    and at the same time find the coeficients using Stochastic approach to see
    if two solutions match
'''
class_1 = np.array([[2,3], [1,2], [2.5, 3.5],
                    [1.5, 2.5], [2,2], [2.5,2.5],
                    [1.5,3],[2.5,1],[1,1]])

class_2 = np.array([[4,5],[5,5],[4,6],
                    [4.5,5.5],[5.5,5.5],[6,6],
                    [5,4],[5,6],[6,5]])

x = np.concatenate(( class_1, class_2))
y = np.array([1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])
y = y.reshape(-1,1)
weights = np.random.rand( x.shape[1], y.shape[1] )
weights /= np.linalg.norm(weights, axis=0)

### Calculate Residual Sum of Squares ###
def RSS(y_true, y_pred):
    return (y_true - y_pred) ** 2

def autograd(y_true, params, inputs, loss=RSS):
    eps = 1e-8
    
    ### Calculate the jacobian of loss w.r.t every params ###
    Jacobian = np.zeros_like(weights)
    for x in range(params.shape[0]):
        for y in range(params.shape[1]):
            params_ = params.copy()
            params_[x][y] += eps

            y_pred_ = inputs @ params_
            y_pred = inputs @ params

            dy = loss(y_true, y_pred_).mean() - loss(y_pred, y_pred).mean()
            dx = eps
            Jacobian[x][y] = dy/dx

    return Jacobian

previous_loss = np.inf
for i  in range(1950):
    predictions = x @ weights
    loss = RSS(y, predictions).mean()

    if(loss > previous_loss or previous_loss - loss < 1e-7):
        print('[*] Early stopping ... ')
        break
    
    previous_loss = loss

    J = -x.transpose() @ (2 * (y - predictions))#autograd(y, weights, x)
    weights -= 1e-3 * J
    print('[*] Epoch {:04d} , RSS is {}'.format(i + 1, loss))

print('-----------------------------------------------------------')
print('[*] Stochastic solutions : ')
print(weights)

### Unique solution calculated method ###
print('[*] Calculated solutions : ')
weights = np.linalg.inv(x.transpose() @ x) @ x.transpose() @ y 
print(weights)
