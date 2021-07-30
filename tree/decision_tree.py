import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.metrics import accuracy_score

from collections import Counter

class Condition:
	def __init__(self, col, val):
		self.col = col 
		self.val = val 
		self.str = ""

	def _is_numeric(self, x):
		if(not isinstance(x, np.ndarray)):
			return isinstance(x, int) or isinstance(x, float)
		return np.issubdtype(x.dtype, np.number) or np.issubdtype(x.dtype, np.number)
	
	def match(self, data):
		col = data[self.col]

		if(self._is_numeric(col)):
			self.str = f'{col} >= {self.val} ?'
			return col >= self.val
		else:
			return col == self.val

	def __str__(self):
		return self.str

	def split(self, data):
		mask = self.match(data)

		true_split = mask[mask == True]
		false_split = mask[mask == False]

		return true_split.index, false_split.index

class Node:
	def __init__(self, condition, true_branch, false_branch):
		self.condition = condition 
		self.true_branch = true_branch 
		self.false_branch = false_branch

class Leaf:
	def __init__(self, classes):
		classes = classes.values 
		class_counts = Counter(classes)

		self.prediction = class_counts.most_common(1)[0][0]


class DecisionTreeClassifier:
	def __init__(self, max_depth=10):
		self.max_depth = max_depth
		self.root = None

	def _get_unique_values(self, class_arr):
		return np.unique(class_arr)

	def _is_numeric(self, x):
		return np.issubdtype(x.dtype, np.number) or np.issubdtype(x.dtype, np.number)

	def _get_class_freq(self, classes):
		unique, counts = np.unique(classes.copy(), return_counts=True)

		return {x:y for x, y in zip(unique, counts)}

	# Criterion based on Gini impurity
	def _get_gini_index(self, classes):
		gini = 1
		for _class in self._get_unique_values(classes):
			p = len(classes[classes == _class])/len(classes)
			gini -= p ** 2

		return gini

	def _get_best_split(self, data, classes, verbose=0):
		columns = set(data.columns)
		best_impurity = 1
		best_condition = None

		for col in columns:
			values = self._get_unique_values(data[col])

			# If this column is a numeric column, values are set
			# as the deciles of the column instead
			if(self._is_numeric(data[col])):
				values = [np.percentile(data[col], i*10) for i in range(1, 10)]

			for val in values:
				condition = Condition(col, val)
				# Get the split mask for true branch and false branch
				true_split, false_split = condition.split(data)

					# Get the ratios of samples in 2 branches
				true_p = len(true_split) / len(data)
				false_p = len(false_split) / len(data)

				# If we cannot partition based on this condition -> skip
				if(len(true_split) == 0 or len(false_split) == 0): continue

				impurity_true = self._get_gini_index(classes[true_split])
				impurity_false = self._get_gini_index(classes[false_split])
				impurity = impurity_true * true_p + impurity_false * false_p 

				if(impurity < best_impurity):
					best_impurity = impurity 
					best_condition = condition 

				if(verbose >= 1):
					print('Splitting by attribute ', col, ' value = ', val, ' impurity = ', impurity)

		return best_impurity, best_condition

	# Tree induction function
	def _build_tree(self, data, classes, current_depth=1, verbose=0):
		best_impurity, best_condition = self._get_best_split(data, classes)
		
		if(best_impurity == 1 or best_condition is None or current_depth >= self.max_depth):
			if(verbose >= 1):
				print(f'[INFO] Reached leaf, current depth {current_depth}, Classes : {self._get_class_freq(classes)}')
			return Leaf(classes)

		true_split, false_split = best_condition.split(data)

		true_data = data.loc[true_split]
		true_classes = classes[true_split]
		true_branch = self._build_tree(true_data, true_classes, current_depth=current_depth+1, verbose=verbose)

		false_data = data.loc[false_split]
		false_classes = classes[false_split]
		false_branch = self._build_tree(false_data, false_classes, current_depth=current_depth+1, verbose=verbose)

		return Node(best_condition, true_branch, false_branch)

	# Train the classification tree
	def fit(self, data, classes, verbose=0):
		self.root = self._build_tree(data, classes, verbose=verbose)

	# Classify function
	def _classify(self, row, node=None):
		if(node is None):
			return None

		if(isinstance(node, Leaf)):
			return node.prediction

		if(node.condition.match(row)):
			return self._classify(row, node.true_branch)
		else:
			return self._classify(row, node.false_branch)

	# A wraper to perform batch classification
	def predict(self, rows):
		labels = []

		for i in range(len(rows)):
			label = self._classify(data.iloc[i], node=self.root)
			labels.append(label)

		return labels


### Load data ###
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
			'marital-status', 'occupation', 'relationship', 'race',
			'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
			'native-country', 'class']

data = pd.read_csv('adult.data', names=columns)

features = data[list(data.columns)[:-1]]
targets = data[list(data.columns)[-1]]
clf = DecisionTreeClassifier(max_depth=10)
clf.fit(features, targets, verbose=1)

predictions = clf.predict(features)
accuracy = accuracy_score(targets, predictions)

print(accuracy)
