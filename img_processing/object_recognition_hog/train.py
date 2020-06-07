import os
import cv2
import pickle
import imutils
import numpy as np

from hog import hog
from svm import KernelSVM
from pca import PCA

from sklearn.decomposition import PCA

# how I think I'm going to do this
# we will need an SVM classifier. But before that
# I'll try to PCA it into 2-element vectors to make it easier

# load images in
DATA_DIR = 'data/'
hog_embeddings = []
labels = []

def train_test_split(x, y, test_size=0.3):
	n_test = int(x.shape[0] * test_size)
	n_train = x.shape[0] - n_test 

	idx = np.array(list(range(x.shape[0])))
	np.random.shuffle(idx)

	idx_train = idx[:n_train]
	idx_test = idx[n_train:x.shape[0]]

	x_train = x[idx_train]
	y_train = y[idx_train]

	x_test = x[idx_test]
	y_test = y[idx_test]

	# print(idx_train, idx_test)
	return x_train, x_test, y_train, y_test

def accuracy_score(predictions, labels):
	accurate_count = 0

	for i in range(len(predictions)):
		if(predictions[i] == labels[i]):
			accurate_count += 1

	accuracy = accurate_count / len(predictions)
	accuracy = "{0:.2f}".format(accuracy)

	return accuracy

if(not os.path.exists("hog_embeddings.pickle") or not os.path.exists("labels.pickle")):
	for (dir, dirs, files) in os.walk(DATA_DIR):
		if(dir != DATA_DIR):
			for file in files:
				label = dir.split("/")[1]

				if(label == 'cat'):
					label = 1
				else:
					label = -1

				labels.append(label)

				abs_path = dir + "/" + file 
				print("[INFO] Reading file : " + abs_path)

				img = cv2.imread(abs_path)
				hog_emb, grad_magnitude = hog(img)

				hog_embeddings.append(hog_emb)


	pca = PCA(n_components = 2)
	print("-----------------------------------------------------------")
	print("[INFO] Implementing Principal component analysis ... ")
	hog_embeddings = np.array(hog_embeddings)

	labels = np.array(labels)
	hog_embeddings = pca.fit_transform(hog_embeddings)


	pickle.dump(hog_embeddings, open("hog_embeddings.pickle", "wb"))
	pickle.dump(labels, open("labels.pickle", "wb"))
else:
	hog_embeddings = pickle.load(open("hog_embeddings.pickle", "rb"))
	labels = pickle.load(open("labels.pickle", "rb"))

print("-----------------------------------------------------------")
print("[INFO] Preparing training phase ... ")
model = KernelSVM()

x_train, x_test, y_train, y_test = train_test_split(hog_embeddings, labels, test_size = 0.2)
model.fit(x_train, y_train, iterations=1000, alpha = 0.1)


### Now validate to model ### 
predictions = model.predict(x_test)
accuracy = accuracy_score(predictions, y_test)
print(accuracy)