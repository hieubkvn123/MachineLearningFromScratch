import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt


from scipy.spatial.distance import cosine
from argparse import ArgumentParser
from sobel import sobel 

parser = ArgumentParser()
parser.add_argument("-i", "--input", required=False, help='path to input image')


args = vars(parser.parse_args())

if(args['input']):
	img = cv2.imread(args['input'])
else:
	img = cv2.imread('img/lenna.png')

# calculates the magnitude over direction histogram
def get_histogram(magnitude, theta):
	hist = np.zeros((9,), dtype=np.float32)

	for column in range(magnitude.shape[0]):
		for row in range(magnitude.shape[1]):
			mag = magnitude[column][row]
			angle = theta[column][row]

			# print(angle)
			lower_bound = min(int(angle/20),8)
			upper_bound = lower_bound + 1

			if(upper_bound == 9):
				upper_bound = 0

			hist[lower_bound] += (abs(upper_bound * 20 - angle)/20) * mag
			hist[upper_bound] += (abs(angle - lower_bound * 20)/20) * mag

	return hist

def hog(img, width = 128, height = 128):
	# resize the image first
	img = cv2.resize(img, (width, height))

	magnitude, theta, mag_cv, angle_cv = sobel(img)

	histograms = np.zeros((int(width/8), int(height/8), 9), dtype=np.float32)
	for column in range(int(width/8)):
		for row in range(int(height/8)):
			mag_grid = magnitude[column*8:(column+1)*8, row*8:(row+1)*8]
			theta_grid = theta[column*8:(column+1)*8, row*8:(row+1)*8]

			histogram = get_histogram(mag_grid, theta_grid)
			histograms[column][row] = histogram

	# now normalize the histograms
	# 128 width and 128 height -> 16 x 16 cells of 8x8 pixels
	hog = np.zeros((histograms.shape[0] - 1, histograms.shape[1] - 1, histograms.shape[2] * 4), dtype=np.float32)
	e = 1e-8 # a small constant to ensure no division by zero

	for column in range(histograms.shape[0] - 1):
		for row in range(histograms.shape[1] - 1):
			grid = histograms[column : column + 2, row:row+2]
			vector = grid.flatten()

			# normalize the individual vectors
			length = np.linalg.norm(vector) + e 

			if(length != 0):
				vector = np.sqrt(vector / length)

			hog[column, row] = vector

	hog = hog.flatten()

	return hog, magnitude


'''
hog, magnitude = hog(img, width=128, height = 128)

# Now we preserve |G| as float to maintain the accuracy 
magnitude = imutils.resize(magnitude, width = 200)

plt.hist(hog, bins = 10)
plt.show()
cv2.imshow("Ahihi", magnitude)
cv2.waitKey(0)
'''