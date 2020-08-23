import numpy as np
import time

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
y = np.array([1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))

def mse(y, y_):
    N = y.shape[0]

    loss = 1/N * (y - y_) ** 2

    return loss

### Use automatic differentiation here ###
def gradient(y,y_,loss=sigmoid):
    h = 1e-8
    return (loss(y, y_ + h) - loss(y, y_)) / h


class Perceptron(object):
    def __init__(self, input_shape=(2,)):
        np.random.seed(int(time.time()))
        self.weights = None
        self.bias = np.random.rand(1)
        self.lr = 1e-3

    def one_hot_encode(self, y):
        return np.eye(self.n_classes)[y]

    def fit(self, x, y):
        if(not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray)):
            print('[INFO] Input and Output must be numpy array')

        if(y.shape[0] != x.shape[0]):
            print('[INFO] Input and Output must be of same length')

        self.input_shape = x.shape[1:]
        self.n_classes = len(np.unique(y))
        self.y = np.array(list(map(self.one_hot_encode, y)))
        self.x = x

        self.output_shape = self.y.shape[1]

        self.weights = np.random.rand(self.x.shape[1], self.y.shape[1])

        for i in range(1000):
            feed_forwarded_input = self.x @ self.weights
            output = sigmoid(feed_forwarded_input)

            l = mse(self.y, output)
            d_1 = gradient(self.y, output, loss=mse)
            d_2 = sigmoid_der(output)
            d_3 = self.x

            g = d_1 * d_2 * d_3

            for j in range(g.shape[0]):
                for i_ in range(self.weights.shape[0]):
                    self.weights[i_] -= self.lr * g[j][i_]

            print('[INFO] Epoch %d, Loss = %.2f ...' % ((i+1),np.sum(l))) 
    
    def predict(self, x):
        return x @ self.weights

    def __str__(self):
        string = "Input shape : %d \n" % self.input_shape
        string += "Output shape : %d \n" % self.output_shape
        string += "Weights : \n"
        string += str(self.weights)

        return string

perceptron = Perceptron(input_shape = x.shape[1])
perceptron.fit(x, y)
print(perceptron.predict(x))
