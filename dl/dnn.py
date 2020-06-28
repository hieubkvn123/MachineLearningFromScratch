import os
import cv2
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt 

class NeuralNet(object):
	def __init__(self):
		self.input = None 
		self.output = None 
		self.theta = None 

	def optimizer(self):