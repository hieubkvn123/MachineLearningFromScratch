import os
import cv2
import imutils

from hog import hog

# how I think I'm going to do this
# we will need an SVM classifier. But before that
# I'll try to PCA it into 2-element vectors to make it easier

