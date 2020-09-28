import numpy as np
import pandas as pd

### For visualization ###
import seaborn as sns
import matplotlib.pyplot as plt 

### utilities ###
from sklearn.preprocessing import LabelEncoder

### We need to predict weights of the fish ###
DATA_FILE = 'Fish.csv'
ITERATIONS = 1000
LR = 1e-3

le = LabelEncoder()
dataframe = pd.read_csv(DATA_FILE, header=0).dropna()
dataframe['Species'] = le.fit_transform(dataframe['Species'])

### Get the most correlated features to Weight ###
num_cols = len(dataframe.columns)
corr = dataframe.corr()['Weight'].sort_values().head(num_cols - 1).tail(4)
columns = list(corr.index)

features = np.array(dataframe[columns].values)
labels = np.array(dataframe['Weight'].values)
labels = labels.reshape(labels.shape[0], 1)

theta = np.random.rand(features.shape[1], labels.shape[1])

def mse(y_true, y_pred):
    e = (y_true - y_pred) ** 2
    return np.mean(e)

def gradient(loss, y_true, features, theta):
    epsilon = 1e-8
    gradient_matrix = []

    y_pred = features @ theta
    loss = mse(y_true, y_pred)

    for i in range(features.shape[0]):
        gradients = []
        for j in range(theta.shape[0]):
            delta = theta + np.eye(theta.shape[0])[j] * epsilon 
            y_pred_ = features @ (theta + delta)
            loss_ = mse(y_true, y_pred_)

            g = (loss_ - loss) * (1/epsilon)
            gradients.append(g)
        gradient_matrix.append(gradients)

    return np.array(gradient_matrix)

for i in range(ITERATIONS):
    y_pred = features @ theta
    e = mse(labels, y_pred)
    print('[INFO] Iteration %d, loss : %.2f' % (i + 1, e))

    g = gradient(mse, labels, features, theta).sum(axis=0)
    print(g)
    theta = theta - LR * g
