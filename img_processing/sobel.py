import cv2
import cmath
import math
import imutils
import numpy as np 
import matplotlib.pyplot as plt

img = np.array(cv2.imread("img/lenna.png"))
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_kernel(grid, kernel):
	return np.sum(np.multiply(grid, kernel))

def get_2d_gaussian(x, y, std):
	# gaussian(x,y,std) = 1/(2*pi*std**2) * exp(-(x**2 + y**2)/(2*std**2))
	return (1/(2*np.pi*std**2)) * np.exp(-(x**2+y**2)/(2*std**2))

def get_gaussian_kernel(k_size=5, std=1):
	center_coord = int((k_size - 1)/2)

	kernel = np.zeros((k_size, k_size), dtype=np.float32)
	for column in range(k_size):
		for row in range(k_size):
			x = abs(row - center_coord)
			y = abs(column - center_coord)
			g = get_2d_gaussian(x,y,std)

			kernel[column, row] = g 

	kernel = np.array(kernel)
	return kernel  

# return gradient magnitude and angle
def sobel(img):
	# initialize the x and y gradient kernel
	G_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
	G_y = G_x.transpose()

	# first is gaussian blurring.
	# basically just convoluting thru a gaussian kernel
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gaussian_kernel = get_gaussian_kernel(k_size=5, std = 1)
	blur = np.zeros((img.shape[0] - 4, img.shape[1] - 4))

	for column in range(img.shape[0] - 4):
		for row in range(img.shape[1] - 4):
			grid = img[column : column + 5, row : row + 5]
			result = int(apply_kernel(grid, gaussian_kernel))
			result = min(result, 255)

			blur[column, row] = result 

	# now calculate the horizontal and vertical gradient
	output_x = np.zeros((blur.shape[0] - 2, blur.shape[1] - 2))

	for column in range(blur.shape[0] - 2):
		for row in range(blur.shape[1] - 2):
			grid = blur[column : column + 3, row : row + 3]
			result = apply_kernel(grid, G_x)
			result = min(result, 255)

			output_x[column, row] = result

	# calculate the vertical gradient
	output_y = np.zeros((blur.shape[0] - 2, blur.shape[1] - 2))
	for row in range(blur.shape[1] - 2):
		for column in range(blur.shape[0] - 2):
			grid = blur[column : column + 3, row : row + 3]
			result = apply_kernel(grid, G_y)
			#result = min(255, result)

			output_y[column, row] = result

	# now calculate the gradient magnitude
	magnitude = np.zeros((blur.shape[0] - 2, blur.shape[1] - 2), dtype=np.uint8)
	threshold = 160
	for column in range(blur.shape[0] - 2):
		for row in range(blur.shape[1] - 2):
			mag = math.sqrt(output_x[column, row]**2 + output_y[column, row]**2)
			mag = int(mag)
			mag = min(threshold, mag)

			magnitude[column, row] = mag

	#print(max(output_x.flatten()))
	#print(max(output_y.flatten()))

	# now find gradient angle theta
	theta = np.zeros((blur.shape[0] - 2, blur.shape[1] - 2), dtype=np.uint8)
	for column in range(blur.shape[0] - 2):
		for row in range(blur.shape[1] - 2):
			# since the angle is in radian
			angle = 0

			#if(output_x[column][row] != 0):
			try:
				angle = (np.arctan2(output_y[column][row] , output_x[column][row]) * (180/np.pi)) % 180
				angle = abs(np.round(angle, 0))
				angle = int(angle)
				#angle = min(angle, 180)
			except: # to catch division by zero
				angle = 0

			theta[column, row] = angle

	mag_cv, angle_cv = cv2.cartToPolar(output_x, output_y, angleInDegrees=True)
	angle_cv = angle_cv - 180

	return magnitude, theta, mag_cv, angle_cv

'''
magnitude, theta, mag_cv,theta_cv = sobel(img)
mag_cv = mag_cv.astype(np.uint8)
print(mag_cv)
print(magnitude)
print(theta_cv[10][20])
print(theta[10][20])

magnitude = imutils.resize(magnitude, width = 500)
theta = imutils.resize(theta, width=500)

#print(theta[np.where(theta > 100)])
cv2.imshow("Gradient magnitude", mag_cv)
cv2.imshow("Gradient direction", theta)
cv2.waitKey(0)
'''