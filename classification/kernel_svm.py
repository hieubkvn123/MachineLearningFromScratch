import cvxopt
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.datasets import make_circles

def generate_linear_seperable_data(num_samples):
    data_1 = np.random.normal(0, 1, size=(num_samples, 2))
    data_2 = np.random.normal(5, 1, size=(num_samples, 2))
    label_1 = np.ones((num_samples,))
    label_2 = np.ones((num_samples,)) * -1

    X = np.concatenate([data_1, data_2])
    Y = np.concatenate([label_1, label_2])

    return X, Y

class KernelSVM:
    def __init__(self, X, Y, kernel='linear', gamma=10.0, C=1.0):
        self.X = X
        self.Y = Y
        self.N = self.X.shape[0]
        self.gamma = gamma
        self.C = C
        self.eps = 1e-5

        self.kernel_name = kernel
        self.kernel = self._get_kernel(kernel)
        self.b = 0.0
        self.alphas = self._get_alphas()
        self.weights = self._get_weights()

    def __gaussian_kernel(self, x, y):
        return np.exp(-1.0*self.gamma*np.dot(np.subtract(x,y).T,np.subtract(x,y)))

    def _get_kernel(self, kernel):
        if(kernel == 'linear'):
            return lambda x, y : np.dot(x, y)
        elif(kernel == 'gaussian'):
            return self.__gaussian_kernel

        return lambda x, y : np.dot(x, y)

    def __get_P(self, x, y):
        M = np.zeros((self.N, self.N))
        Y = np.outer(y, y)

        for i in range(self.N):
            for j in range(self.N):
                x_ij = self.kernel(x[i], x[j])
                M[i, j] = x_ij

        return np.multiply(Y, M)

    def _get_alphas(self):
        P = self.__get_P(self.X, self.Y)
        q = np.ones((self.N,)) * -1
        G = np.diag(np.ones(self.N) * -1)
        h = np.zeros(self.N)
        A = self.Y.reshape(1, self.N).astype('float')
        b = 0.0

        P = cvxopt.matrix(P)
        q = cvxopt.matrix(q)
        G = cvxopt.matrix(G)
        h = cvxopt.matrix(h)
        A = cvxopt.matrix(A)
        b = cvxopt.matrix(b)

        alphas = np.ravel(cvxopt.solvers.qp(P, q, G, h, A, b)['x'])

        self.alphas = alphas
        support_vectors = self.X[alphas > self.eps] # x*
        support_labels  = self.Y[alphas > self.eps] # y*

        for i in range(len(support_vectors)):
            w_T_x = 0
            for j in range(self.N):
                w_T_x += alphas[j] * self.Y[j] * self.kernel(self.X[j], support_vectors[i])
            self.b += support_labels[i] - w_T_x

        self.b /= support_vectors.shape[0]
        return alphas

    def _get_weights(self): # only apply when kernel is linear
        if(self.kernel_name == 'linear'):
            return np.sum(self.X*self.Y.reshape(self.N,1)*self.alphas.reshape(self.N,1),axis=0,keepdims=True)
        else:
            return None

    def predict(self, x):
        w_T_x = 0.0

        if(self.kernel_name != 'linear'):
            for i in range(self.N):
                w_T_x += self.alphas[i] * self.Y[i] * self.kernel(self.X[i], x)
        else:
            w_T_x = np.dot(self.weights, x)[0]
        pred = w_T_x + self.b
        return np.sign(pred)

    def predict_batch(self, x):
        y_hat = np.array([self.predict(x_) for x_ in x])
        return y_hat

    def plot_decision_boundary(self):
        fig, ax = plt.subplots()

        x_min, x_max = self.X[:,0].min() - 1, self.X[:,0].max() + 1
        y_min, y_max = self.X[:,1].min() - 1, self.X[:,1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

        Z = self.predict_batch(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z)
        for label in np.unique(self.Y):
            data = self.X[self.Y == label]
            ax.scatter(data[:,0], data[:,1])

        plt.show()

X, Y = generate_linear_seperable_data(50)
svm = KernelSVM(X, Y, kernel='linear')
svm.plot_decision_boundary()

X, Y = make_circles(n_samples=100, noise=0.1, factor=0.1, random_state=1)
Y[Y == 0] = -1
svm = KernelSVM(X, Y, kernel='gaussian')
svm.plot_decision_boundary()
