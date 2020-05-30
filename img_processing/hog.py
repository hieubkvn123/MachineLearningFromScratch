import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt

from sobel import sobel 

img = cv2.imread("lenna.png")

# calculates the magnitude over direction histogram
counter = 0
def get_histogram(magnitude, theta):
	global counter 
	counter += 1

	if(counter == 1):
		np.savetxt("magnitude.txt", magnitude)
		np.savetxt("theta.txt", theta)

	hist = np.zeros((9,), dtype=np.float32)

	for column in range(magnitude.shape[0]):
		for row in range(magnitude.shape[1]):
			mag = magnitude[column][row]
			angle = theta[column][row]

			# print(angle)
			lower_bound = int(angle/20)
			upper_bound = int(angle/20) + 1 

			if(upper_bound == 9):
				upper_bound = 0

			hist[lower_bound] += (upper_bound * 20 - angle)/20 * mag
			hist[upper_bound] += (angle - lower_bound * 20)/20 * mag

	if(counter == 1):
		np.savetxt("hist.txt", hist)
	return hist

def hog(img, width = 128, height = 128):
	# resize the image first
	img = cv2.resize(img, (width, height))

	magnitude, theta = sobel(img)

	histograms = np.zeros((int(width/8), int(height/8), 9), dtype=np.float32)
	for column in range(int(width/8)):
		for row in range(int(height/8)):
			mag_grid = magnitude[column*8:(column+1)*8, row*8:(row+1)*8]
			theta_grid = theta[column*8:(column+1)*8, row*8:(row+1)*8]

			histogram = get_histogram(mag_grid, theta_grid)
			histograms[column][row] = histogram

	# now normalize the histograms
	# 128 width and 128 height -> 16 x 16 cells of 8x8 pixels
	hog = np.zeros((histograms.shape[0] - 1, histograms.shape[1] - 1, histograms.shape[2] * 4))

	for column in range(histograms.shape[0] - 1):
		for row in range(histograms.shape[1] - 1):
			grid = histograms[column : column + 2, row:row+2]
			vector = grid.flatten()

			# normalize the individual vectors
			length = np.sqrt(np.sum(vector * vector))

			if(length != 0):
				vector = vector / length

			hog[column, row] = vector

	hog = hog.flatten()

	return hog, magnitude


hog, magnitude = hog(img, width=64, height = 128)
#hog = cv2.HOGDescriptor().compute(img)
# print(hog)
plt.hist(hog, bins = 10)
plt.show()