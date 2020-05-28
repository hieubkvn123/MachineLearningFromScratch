import cv2
import math
import imutils
import numpy as np 
import matplotlib.pyplot as plt

img = np.array(cv2.imread("lenna.png"))

G_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
G_y = G_x.transpose()

# preprocess the image, gaussian blur it first
blur = cv2.GaussianBlur(img, (5,5), 0)
blur = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)


# now perform horizontal convolution
# and vertical convolution on it
output_x = np.zeros((img.shape[0] - 2, img.shape[1] - 2))
def apply_kernel(img, kernel):
	return np.sum(np.multiply(img, kernel))

counter = 0 # #operations = (H - 2) * (W - 2) for 3x3 kernel
# print(blur)
for column in range(img.shape[0]-2):
	for row in range(img.shape[1]-2):
		counter += 1 # to check number of convolution operation
		# grab a 3x3 grid and multiply
		grid = blur[column : column + 3, row : row + 3]

		result = int(apply_kernel(grid, G_x))
		result = min(255, result)
		output_x[column, row] = result

# compare to opencv Sobel
output_x_cv = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=1)

# now perform vertical convolution
output_y = np.zeros((img.shape[0] - 2, img.shape[1] - 2))

counter = 0
for row in range(img.shape[1] - 2):
	for column in range(img.shape[0] - 2):
		counter += 1

		grid = blur[column : column + 3, row : row + 3]

		result = int(apply_kernel(grid, G_y))
		result = min(255, result)
		output_y[column, row] = result

print(max(output_y.flatten()))

# compare to opencv Sobel
output_y_cv = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=1)

# calculate gradient magnitude 
magnitude = np.zeros((img.shape[0] - 2, img.shape[0] - 2), dtype=np.uint8)
threshold = 70

for column in range(img.shape[0] - 2):
    for row in range(img.shape[1] - 2):
        mag = math.sqrt(output_x[column][row]**2 + output_y[column][row]**2)
        mag = int(mag)
        mag = max(mag, threshold)
        magnitude[column][row] = mag

magnitude = imutils.resize(magnitude, width = 500)
output_x_cv = imutils.resize(output_x_cv, width = 500)
output_y_cv = imutils.resize(output_y_cv, width = 500)
output_x = imutils.resize(output_x, width=500)
output_y = imutils.resize(output_y, width=500)

cv2.imshow("Gradient Magnitude", magnitude)

cv2.imshow("Gradient X opencv", output_x_cv)
cv2.imshow("Gradient X", output_x)

cv2.imshow("Gradient Y opencv", output_y_cv)
cv2.imshow("Gradient Y", output_y)

cv2.waitKey(0)
