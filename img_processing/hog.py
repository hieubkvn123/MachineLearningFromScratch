import cv2
import imutils

from sobel import sobel 

img = cv2.imread("lenna.png")

def get_histogram(magnitude, theta):
	hist = np.zeros((9,1))

	for column in range(magnitude.shape[0]):
		for row in range(magnitude.shape[1]):
			mag = magnitude[column][row]
			angle = theta[column][row]

			lower_bound = int(angle/20)
			upper_bound = int(angle/20) + 1

			if(upper_bound == 9):
				upper_bound = 0

			hist[lower_bound] = (upper_bound * 20 - angle)/20 * magnitude
			hist[upper_bound] = (angle - lower_bound * 20)/20 * magnitude

	return hist

def hog(img, width = 128, height = 128):
	# resize the image first
	img = cv2.resize(img, (width, height))

	magnitude, theta = sobel(img)

	

	return magnitude, theta


magnitude, theta = hog(img, width=256, height = 256)

cv2.imshow("Magnitute", magnitude)
cv2.waitKey(0)