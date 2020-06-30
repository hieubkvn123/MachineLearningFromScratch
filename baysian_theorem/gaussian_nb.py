import numpy as np

class GaussianNB(object):
    def __init__(self):
        self.x = []
        self.y = []

        self.N  = 0 

    def fit(self, x, y):
    	if(x.shape[0] != y.shape[0]):
    		raise Exception("[INFO] Input and output must have the same length ... ")

		if(len(x.shape) < 2):
			raise Exception("[INFO] Input must be an array of vectors ...")
			
        self.x = x 
        self.y = y
		self.N = x.shape[0]
		
	def get_mean(self):
		

