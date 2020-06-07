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
from svm import KernelSVM 
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