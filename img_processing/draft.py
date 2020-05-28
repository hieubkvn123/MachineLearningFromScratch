import numpy as np 
import math

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

	return kernel

print(get_gaussian_kernel())