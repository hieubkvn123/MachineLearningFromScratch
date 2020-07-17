from layers import Dense
from dnn import NeuralNet
from optimizers import SGD
from losses import MSE
import numpy as np


class_1 = np.array([[2,3], [1,2], [2.5, 3.5],
                    [1.5, 2.5], [2,2], [2.5,2.5],
                    [1.5,3],[2.5,1],[1,1]])

class_2 = np.array([[4,5],[5,5],[4,6],
                    [4.5,5.5],[5.5,5.5],[6,6],
                    [5,4],[5,6],[6,5]])

x_train = np.concatenate((class_1, class_2))
y_train = np.array([1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])
#dense = Dense(24)
# a = np.array([[1,2,3], [2,4,6]])
#b = dense(a)

net = NeuralNet(input_dim=x_train.shape[1])
net.add_layer(Dense(4))
net.add_layer(Dense(2))

optimizer = SGD(learning_rate=0.000000001)
loss = MSE()
net.compile(optimizer, loss)
net.fit(x_train, y_train)

# print(b)
print(net(x_train))
for output in net(x_train):
    print(np.argmax(output))
