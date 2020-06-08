import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt 

class NeuralNet(object):
	def __init__(self):
		self.input = None 
		self.output = None 
		self.theta = None 

	def optimizer(self, algo='sgd', beta_1=0.9,beta_2=0.999):
		operation = lambda g, w, alpha : w - alpha * g

		def adam(g, w, i, m_t, g_t, alpha):
			m_t = beta_1 * m_t + (1-beta_1)*g 
			v_t = beta_2 * v_t + (1-beta_2)*(g**2)

			m_t_hat = m_t/(1-beta_1**i)
			v_t_hat = v_t/(1-beta_2**i)

			w = m_t_hat/(math.sqrt(v_t_hat) + 1e-08)

			return w, m_t, v_t 

		if(algo == 'adam'):
			operation = lambda g,  

	def fit(self, x, y, lr=1e-3, optimizer='sgd'):
