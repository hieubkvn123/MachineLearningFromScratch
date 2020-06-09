import os
import cv2
import pickle
import imutils
import numpy as np

from hog import hog
from svm import KernelSVM
from pca import PCA

from sklearn.decomposition import PCA
from sklearn.svm import SVC

# how I think I'm going to do this
# we will need an SVM classifier. But before that
# I'll try to PCA it into 2-element vectors to make it easier

# load images in
DATA_DIR = 'data/'
hog_embeddings = []
labels = []

hog_embeddings_val = []
labels_val = []

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

pca = PCA(n_components = 4)
print("[INFO] Reading training data ... ")
print("-------------------------------------------------------------")

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

print("-------------------------------------------------------------")
print("[INFO] Reading validation data")

VAL_DIR = 'validation/'
if(not os.path.exists("hog_embeddings_val.pickle") or not os.path.exists("labels_val.pickle")):
	for (dir, dirs, files) in os.walk(VAL_DIR):
		if(dir != VAL_DIR):
			for file in files:
				label_val = dir.split("/")[1]
				if(label_val == 'cat'):
					label_val = 1
				else:
					label_val = -1

				labels_val.append(label_val)

				abs_path = dir + "/" + file 
				print("[INFO] Reading file : " + abs_path)

				img = cv2.imread(abs_path)
				hog_emb, grad_magnitude = hog(img)

				hog_embeddings_val.append(hog_emb)

	print("-----------------------------------------------------------")
	print("[INFO] Implementing Principal component analysis ... ")
	hog_embeddings_val = np.array(hog_embeddings_val)

	labels_val = np.array(labels_val)
	hog_embeddings_val = pca.transform(hog_embeddings_val)


	pickle.dump(hog_embeddings_val, open("hog_embeddings_val.pickle", "wb"))
	pickle.dump(labels_val, open("labels_val.pickle", "wb"))
else:
	hog_embeddings_val = pickle.load(open("hog_embeddings_val.pickle", "rb"))
	labels_val = pickle.load(open("labels_val.pickle", "rb"))

print("-----------------------------------------------------------")
print("[INFO] Preparing training phase ... ")
model = KernelSVM()


### Training phase ###
model.fit(hog_embeddings, labels, iterations=15000, alpha = 0.0001)


### Now validate to model ### 
### Kind of failed :((( ###
predictions = model.predict(hog_embeddings_val)
convert = lambda p : 'cat' if p == 1 else 'dog'

accuracy = accuracy_score(predictions, labels_val)
predictions = [convert(x) for x in predictions]

print("[INFO] Home made recipe : " + str(accuracy))

# Comparing to an sklearn model
model = SVC(kernel='rbf')
model.fit(hog_embeddings, labels)
predictions = model.predict(hog_embeddings_val)

accuracy = accuracy_score(predictions, labels_val)
print("[INFO] sklearn's recipe : " + str(accuracy))
