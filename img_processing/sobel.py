import cv2
import math
import imutils
import numpy as np 
import matplotlib.pyplot as plt

img = np.array(cv2.imread("lenna.png"))
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

G_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
G_y = G_x.transpose()

# preprocess the image, gaussian blur it first
# these gaussian blurred versions will be used in opencv
# for comparison purpose
blur_ = cv2.GaussianBlur(img, (5,5), 0)
blur_ = cv2.cvtColor(blur_, cv2.COLOR_BGR2GRAY)

def apply_kernel(img, kernel):
	return np.sum(np.multiply(img, kernel))

def get_gaussian(x,y,std):
	g = 1/(2*np.pi*(std**2)) * np.exp(-(x**2 + y**2)/(2*(std**2)))

	return g

def get_gaussian_kernel(k_size=5, std=1):
	kernel = np.zeros((k_size, k_size), dtype=np.float32)
	center_coord = int((k_size - 1)/2)

	for column in range(kernel.shape[0]):
		for row in range(kernel.shape[1]):
			x = abs(center_coord - row)
			y = abs(center_coord - column)

			g = get_gaussian(x,y,std)
			kernel[column, row] = g

	kernel = np.array(kernel)
	return kernel


### manually gaussian blurring the image ###
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = np.zeros((img.shape[0] - 4, img.shape[0] - 4))
kernel = get_gaussian_kernel()
for column in range(img.shape[0] - 4):
	for row in range(img.shape[1] - 4):
		grid = img[column : column + 5, row : row + 5]

		result = int(apply_kernel(grid, kernel))
		blur[column, row] = result

# now perform horizontal convolution
# and vertical convolution on it
output_x = np.zeros((blur.shape[0] - 2, blur.shape[1] - 2))
counter = 0 # #operations = (H - 2) * (W - 2) for 3x3 kernel
# print(blur)
for column in range(blur.shape[0]-2):
	for row in range(blur.shape[1]-2):
		counter += 1 # to check number of convolution operation
		# grab a 3x3 grid and multiply
		grid = blur[column : column + 3, row : row + 3]

		result = int(apply_kernel(grid, G_x))
		result = min(255, result)
		output_x[column, row] = result

# compare to opencv Sobel
output_x_cv = cv2.Sobel(blur_, cv2.CV_32F, 1, 0, ksize=1)




### Now calculate G_y (Vertical gradient) ###



# now perform vertical convolution
output_y = np.zeros((blur.shape[0] - 2, blur.shape[1] - 2))
counter = 0
for row in range(blur.shape[1] - 2):
	for column in range(blur.shape[0] - 2):
		counter += 1

		grid = blur[column : column + 3, row : row + 3]

		result = int(apply_kernel(grid, G_y))
		result = min(255, result)
		output_y[column, row] = result

# compare to opencv Sobel
output_y_cv = cv2.Sobel(blur_, cv2.CV_32F, 0, 1, ksize=1)

# calculate gradient magnitude 
magnitude = np.zeros((blur.shape[0] - 2, blur.shape[0] - 2), dtype=np.uint8)
threshold = 70

for column in range(blur.shape[0] - 2):
    for row in range(blur.shape[1] - 2):
        mag = math.sqrt(output_x[column][row]**2 + output_y[column][row]**2)
        mag = int(mag)
        mag = min(mag, threshold)
        magnitude[column][row] = mag





### Visualizing the result ###




magnitude = imutils.resize(magnitude, width = 500)
output_x_cv = imutils.resize(output_x_cv, width = 500)
output_y_cv = imutils.resize(output_y_cv, width = 500)
output_x = imutils.resize(output_x, width=500)
output_y = imutils.resize(output_y, width=500)

cv2.imshow("Gradient X opencv", output_x_cv)
cv2.imshow("Gradient X", output_x)
cv2.imshow("Gradient Y opencv", output_y_cv)
cv2.imshow("Gradient Y", output_y)
cv2.imshow("Gradient Magnitude", magnitude)

cv2.waitKey(0)
