import numpy as np 
import pandas as pd 

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from svm import KernelSVM 

from pca import pca

le = LabelEncoder()
ss = StandardScaler()

data_url = 'https://raw.githubusercontent.com/hieubkvn123/data/master/bank_risk.csv'
raw_data = pd.read_csv(data_url, header = 0).dropna()

raw_data['Gender'].replace(['Male', 'Female'], [1,0], inplace=True)
raw_data['Married'].replace(['Yes', 'No'],[1,0], inplace=True)
raw_data['Education'].replace(['Graduate', 'Not Graduate'], [1,0], inplace=True)
raw_data['outcome'].replace(['Y', 'N'], [1,-1], inplace=True)

data = raw_data[['Gender', 'Married', 'Education', 'ApplicantIncome', 'LoanAmount', 'Credit_History']]

x = data.to_numpy()
x = ss.fit_transform(x)
x = pca(x)
y = raw_data['outcome'].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

model = KernelSVM()
model.fit(x_train,y_train, alpha = 0.01, iterations=200)


predictions = model.predict(x_test)
accuracy = accuracy_score(predictions, y_test)

print(accuracy)
