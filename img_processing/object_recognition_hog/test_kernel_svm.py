import math
import numpy as np 
import pandas as pd 

### Not gonna use these for long, just for validation purpose ###
'''
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
'''

### NONONONO ###

### self-written machine learning, kinda slow but good enuff ###
# from svm import KernelSVM 
from pca import PCA

# now the moment of truth, comparing to sklearn
from sklearn.svm import SVC 

def accuracy_score(predictions, labels):
	accurate_count = 0

	for i in range(len(predictions)):
		if(predictions[i] == labels[i]):
			accurate_count += 1

	accuracy = accurate_count / len(predictions)
	accuracy = "{0:.2f}".format(accuracy)

	return accuracy

def standardize(x):
	if(not isinstance(x, np.ndarray)):
		print("[INFO] x must be a vector ... ")
		return None
	else:
		if(len(x.shape) != 1):
			print('[INFO] x must be a flat vector ... ')
			return None 
		else:
			v_length = np.linalg.norm(x)
			x_std = x/v_length

			return x_std

def train_test_split(x, y, test_size=0.3):
	n_test = int(x.shape[0] * test_size)
	n_train = x.shape[0] - n_test 

	idx = np.array(list(range(x.shape[0])))
	np.random.shuffle(idx)

	idx_train = idx[:n_train]
	idx_test = idx[n_train:x.shape[0]]

	x_train = x[idx_train]
	y_train = y[idx_train]

	x_test = x[idx_test]
	y_test = y[idx_test]

	# print(idx_train, idx_test)
	return x_train, x_test, y_train, y_test


# Note, this is for binary classification only
def kernel(kernel='linear', gamma = 0.01, r=1/2, d=2):
	'''
		gamma : 1/2*sigma**2 for the gaussian kernel
		r : the bias for polynomial kernel
		d : the power base for the polynomial kernel
	'''

	# the default linear kernel
	operation = lambda x, y : np.dot(x, y)

	if(kernel == 'rbf'):
		operation = lambda x, y : -gamma*(np.linalg.norm(x-y)**2)
	if(kernel == 'poly'):
		operation = lambda x, y : (np.dot(x,y) + r) ** d 

	return operation

# mse loss
def loss(predictions, labels):
	if(predictions.shape[0] != labels.shape[0]):
		print("[INFO] Labels and Predictions are not of the same shape ... ")
		return None
	else:
		# loop thru pairs of prediction and label
		loss = 0
		for j in range(predictions.shape[0]): # or labels.shape[0]
			loss += (predictions[j] - labels[j]) ** 2

		return 1/predictions.shape[0] * math.sqrt(loss)

class KernelSVM:
	def __init__(self):
		self.x = None 
		self.y = None 
		self.w = None
		self.bias = None
		self.kernel = None

		self.m_t = None
		self.v_t = None
		self.m_t_v = None
		self.v_t_v = None

		self.beta_1 = 0.9
		self.beta_2 = 0.999

	def fit(self,x, y, iterations=100000, alpha=0.0001, l=0.01):
		'''
			- iterations is the max number of training iterations
			- alpha is the learning rate
			- l is the regularization parameter
		'''

		# just create a copy for inference purpose
		self.x = x
		self.y = y

		bias = 10.0

		if(not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray)):
			print("[INFO] The input and output must be numpy array ... ")
			return None
		else:
			# check if the input is of right dimension
			if(len(x.shape) != 2 or len(y.shape) != 1):
				print("[INFO] Input and output are of wrong shape ...")
				return None
			else:
				# svm is optimized based on hinge loss
				# hinge(x,w) = Max(0, 1 - y * <x,w>)
				# we need to optimize 2/||w|| -> the regularized loss is 
				# L = summation(Max(0, 1 - y * <x,w>)) + ||w||**2/2
				K = kernel(kernel='rbf')
				self.kernel = K

				# first, calculate all f_i vectors from the kernel
				dataset_size = x.shape[0]
				f_i = np.zeros((dataset_size, dataset_size))

				# construct f_i which is a matrix of kernelized inputs
				for i in range(dataset_size):
					f = np.zeros((dataset_size,))
					for j in range(dataset_size):
						f[j] = K(x[i], x[j])

					f_i[i] = f 

				# print(f_i)

				# now all we have to do is applying svm on f_1 and y
				self.w = np.ones((dataset_size, ), dtype=np.float32) # a vector with same size as f_i input vectors
				self.m_t = np.zeros((dataset_size, ), dtype=np.float32) # for optimization
				self.v_t = np.zeros((dataset_size, ), dtype=np.float32)

				# bias corrected version of m_t and v_t
				self.m_t_v = np.zeros((dataset_size, ), dtype=np.float32)
				self.v_t_v = np.zeros((dataset_size, ), dtype=np.float32)

				# loop thru the training iterations
				previous_loss = 0
				for i in range(iterations) :
					# alpha = alpha * (decay_rate * 1/(i+1))
					predictions = np.zeros((y.shape[0], ))
					# loop thru all the data
					for j in range(f_i.shape[0]):
						f_i_ = f_i[j]
						prediction = np.dot(self.w, f_i_) + bias 
						predictions[j] = prediction

						# check if this is a mis-classification case
						if(1 - y[j] * (np.dot(self.w, f_i_) + bias) > 0): # misclassification
							# loop thru the elements in the weight vector
							for i_ in range(self.w.shape[0]):
								gradient = -y[j] * f_i_[i_] + l * self.w[i_]
								self.m_t[i_] = self.beta_1*self.m_t[i_] + (1 - self.beta_1) * gradient
								self.v_t[i_] = self.beta_2*self.v_t[i_] + (1 - self.beta_2) * (gradient**2)

								# bias correction
								self.m_t_v[i_] = self.m_t[i_]/(1 - self.beta_1 ** (i+1))
								self.v_t_v[i_] = self.v_t[i_]/(1 - self.beta_2 ** (i+1))

								self.w[i_] = self.w[i_] - alpha * (self.m_t_v[i_]/(math.sqrt(self.v_t_v[i_]) + 1e-6))# (-y[j] * f_i_[i_] + l * self.w[i_])

							bias = bias + y[j] * alpha

						else: # no misclassification occured
							# loop thru the elements of the weight vector again 
							for i_ in range(self.w.shape[0]):
								gradient = l*self.w[i_]

								self.m_t[i_] = self.beta_1*self.m_t[i_] + (1 - self.beta_1) * gradient
								self.v_t[i_] = self.beta_2*self.v_t[i_] + (1 - self.beta_2) * (gradient**2)

								# bias correction
								self.m_t_v[i_] = self.m_t[i_]/(1 - self.beta_1 ** (i+1))
								self.v_t_v[i_] = self.v_t[i_]/(1 - self.beta_2 ** (i+1))

								self.w[i_] = self.w[i_] - alpha * (self.m_t_v[i_]/(math.sqrt(self.v_t_v[i_]) + 1e-6))# (-y[j] * f_i_[i_] + l * self.w[i_])

					mse = loss(predictions, y)

					if(mse > previous_loss and i > 10000):
						break

					# if the reduction in loss barely matters
					# we break the process
					if(previous_loss - mse < 1e-8 and i > 10000):
						break

					previous_loss = mse
					print("[INFO] Epoch : " + str(i+1) + " | Loss = " + "{0:.2f}".format(mse))
		self.bias = bias

	def predict(self, x):
		predictions = []

		# for each of the new data
		for i in range(x.shape[0]):
			# loop thru the training dataset
			prediction = 0
			for j in range(self.x.shape[0]):
				prediction += self.y[j] * self.w[j] * self.kernel(x[i], self.x[j])

			prediction = prediction + self.bias

			predictions.append(np.sign(prediction))

		return predictions 

# ss = StandardScaler()

data_url = 'https://raw.githubusercontent.com/hieubkvn123/data/master/bank_risk.csv'
raw_data = pd.read_csv(data_url, header = 0).dropna()

raw_data['Gender'].replace(['Male', 'Female'], [1,0], inplace=True)
raw_data['Married'].replace(['Yes', 'No'],[1,0], inplace=True)
raw_data['Education'].replace(['Graduate', 'Not Graduate'], [1,0], inplace=True)
raw_data['outcome'].replace(['Y', 'N'], [1,-1], inplace=True)

data = raw_data[['Gender', 'Married', 'Education', 'ApplicantIncome', 'LoanAmount', 'Credit_History']]

x = data.to_numpy()
# x = ss.fit_transform(x)

print("[INFO] Standardizing input vectors ... ")
for i in range(x.shape[0]):
	# standardize each vector
	x[i] = standardize(x[i])

print("[INFO] Implementing principal components analysis ... ")
pca = PCA()
x = pca.fit(x)
y = raw_data['outcome'].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

model = KernelSVM()
model.fit(x_train,y_train, alpha = 0.01, iterations=100)

predictions = model.predict(x_test)
accuracy = accuracy_score(predictions, y_test)

print("[INFO] Home made recipe : " + str(accuracy))

model = SVC(kernel='rbf')
model.fit(x_train, y_train)

predictions = model.predict(x_test)
accuracy = accuracy_score(predictions, y_test)
print("[INFO] Sklearn's recipe : " + str(accuracy))