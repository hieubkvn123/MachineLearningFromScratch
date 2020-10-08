import numpy as np
import matplotlib.pyplot as plt

from data_loader import train_images, train_labels
from data_loader import test_images, test_labels
from data_loader import train_labels_original, test_labels_original

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

BATCH_SIZE=60

class Discriminator():
    def __init__(self):
        self.weights = np.random.rand(train_images.shape[1], train_labels.shape[1])
        self.penalty = 0.3
        self.batch   = 32
        self.eps = 1e-8
        self.lr = 1e-3
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.momentum = 0
        self.velocity = 0

    def _softmax(self, x):
        exp_x = np.exp(x)
        sum_exp = np.expand_dims(exp_x.sum(axis=1), axis=1)

        softmax = exp_x / sum_exp

        return softmax

    def softmax_crossentropy(self, targets, predictions):
        loss = -targets * np.log(predictions)
        loss = np.sum(loss, axis=1)
        return loss

    def train_step(self, i, targets, features):
        i = i + 1
        X = features
        predictions = self.forward(features, targets)
        
        loss = self.softmax_crossentropy(targets, predictions)
        print(loss.mean())

        gradients = - (X.transpose() @ (targets - predictions))
        self.momentum = self.beta_1 * self.momentum + (1 - self.beta_1) * gradients
        self.velocity = self.beta_2 * self.velocity + (1 - self.beta_2) * (gradients ** 2)

        momentum_ = self.momentum / (1 - self.beta_1 ** i)
        velocity_ = self.velocity / (1 - self.beta_2 ** i)

        # self.weights = self.weights - self.lr * gradients
        self.weights = self.weights - self.lr * momentum_/(np.sqrt(velocity_) + self.eps)


    def forward(self, x, y):
        x = x / np.expand_dims(np.linalg.norm(x, axis=1), axis=1)
        W = self.weights / np.expand_dims(np.linalg.norm(self.weights, axis=0), axis=0)

        logits = x @ W 

        theta = np.arccos(np.clip(logits, -1.0 + self.eps, 1.0 - self.eps))
        marginal_logit = np.cos(theta + self.penalty)
        logits = logits + (marginal_logit - logits) * y

        logits *= 30
        out = self._softmax(logits)

        return out

    def train(self, trainX, trainY, epochs=200):
        num_batches = trainX.shape[0] // self.batch + 1
        for i in range(epochs):
            for j in range(num_batches):
                X = trainX[j*self.batch:(j+1)*self.batch]
                Y = trainY[j*self.batch:(j+1)*self.batch]
                self.train_step(i, Y, X)

disc = Discriminator()
disc.train(train_images, train_labels)

X = train_images @ disc.weights

X_ = test_images @ disc.weights

svm = SVC(kernel='rbf', C=1.0, gamma='auto', probability=True)
svm.fit(X, train_labels_original)

predictions = svm.predict(X_)
accuracy = accuracy_score(predictions, test_labels_original)
