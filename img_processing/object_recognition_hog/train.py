import os
import cv2
import imutils
import numpy as np

from hog import hog
from svm import KernelSVM
from pca import pca

from sklearn.decomposition import PCA

# how I think I'm going to do this
# we will need an SVM classifier. But before that
# I'll try to PCA it into 2-element vectors to make it easier

# load images in
DATA_DIR = 'data/'
hog_embeddings = []
labels = []

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


print("-----------------------------------------------------------")
print("[INFO] Preparing training phase ... ")
model = KernelSVM()
print(labels)
model.fit(hog_embeddings, labels, iterations=100)

