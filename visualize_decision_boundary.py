import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X_MIN = -5
X_MAX = 5
class_1 = np.random.normal(0,0.5, size=(50, 2))
class_2 = np.random.normal(3,0.5, size=(50, 2))
class_1 = np.clip(class_1, X_MIN, X_MAX)
class_2 = np.clip(class_2, X_MIN, X_MAX)

### Visualize the mesh ###
X_MIN = np.concatenate([class_1[:,0], class_2[:,0]]).min() - 2
X_MAX = np.concatenate([class_1[:,0], class_2[:,0]]).max() + 2

Y_MIN = np.concatenate([class_1[:,1], class_2[:,1]]).min() - 2 
Y_MAX = np.concatenate([class_1[:,1], class_2[:,1]]).max() + 2

xx, yy = np.meshgrid(np.arange(X_MIN, X_MAX, 0.02),
                     np.arange(Y_MIN, Y_MAX, 0.02))
model = LinearSVC().fit(np.concatenate([class_1, class_2]),
                    np.concatenate([np.ones(shape=(50,)), -np.ones(shape=(50,))]))

z = model.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

print(accuracy_score(model.predict(np.concatenate([class_1, class_2])), 
      np.concatenate([np.ones(shape=(50,)), -np.ones(shape=(50,))])))
plt.pcolormesh(xx, yy, z)

plt.scatter(class_1[:,0], class_1[:,1], color='blue', alpha=0.5)
plt.scatter(class_2[:,0], class_2[:,1], color='orange', alpha=0.5)
plt.show()
