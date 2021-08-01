import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from collections import Counter

class Condition:
	def __init__(self, col, val):
		'''
			Splits a pandas data frame based on a condition applied on one of
			its column. For numeric columns, the condition will match when the
			value of that column is greater than or equal to self.val

			Args :
				col : Column name in the dataframe
				val : The split value. equality condition is applied when the column
					  is categorical. greater than or equal to condition will be applied
					  if the column is numeric
		'''
		self.col = col 
		self.val = val 
		self.str = ""

	def __str__(self):
		return self.str

	def _is_numeric(self, x):
		'''
			Checks if a value or a np.ndarray is of numeric type
			Args :
				x : A scalar or a np.ndarray vector
		'''
		if(not isinstance(x, np.ndarray)):
			return isinstance(x, int) or isinstance(x, float)
		return np.issubdtype(x.dtype, np.number) or np.issubdtype(x.dtype, np.number)
	
	def match(self, data):
		'''
			Returns a boolean mask which is True for rows that matches the condition of this
			object, False otherwise/

			Args :
				data : A pd.DataFrame to match against the condition
		'''
		col = data[self.col]

		if(self._is_numeric(col)):
			self.str = f'{col} >= {self.val} ?'
			return col >= self.val
		else:
			return col == self.val

	def split(self, data):
		'''
			Returns two index Series of the rows that match and do not match the condition
			Args :
				data : A pd.DataFrame to be splitted
		'''
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
		if(isinstance(classes, pd.Series)):
			classes = classes.values 
		class_counts = Counter(classes)

		self.classes = classes 
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
		
		if(best_condition is None or current_depth >= self.max_depth or len(np.unique(classes)) == 1):
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

	# Function for merging two leaf branch to one leaf branch
	def __merge_leaf(self, node):
		if(isinstance(node.true_branch, Leaf) and isinstance(node.false_branch, Leaf)):
			classes_true = node.true_branch.classes  
			classes_false = node.false_branch.classes 
			new_leaf = Leaf(np.concatenate([classes_true, classes_false]))

			return new_leaf
		else: 
			return None

	def __trim_leaf(self, node, data, classes, verbose=0):
		if(isinstance(node.true_branch, Leaf) and isinstance(node.false_branch, Leaf)):
			leaf = self.__merge_leaf(node)
			err_leaf = sum((leaf.prediction != classes).astype('int'))

			pred_no_leaf = []
			for i in range(len(data)):
				label = self._classify(data.iloc[i], node=node)
				pred_no_leaf.append(label)

			err_no_leaf = sum((np.array(pred_no_leaf) != classes).astype('int'))

			if(err_leaf <= err_no_leaf):
				if(verbose >= 1):
					print(f'[INFO] Tree pruned, errors without merge : {err_no_leaf}, errors with merge : {err_leaf}')
				return leaf 
			else:
				return node 
		else:
			return node

	# Post-pruning
	def _post_pruning(self, node, data, classes, verbose=0):
		if(isinstance(node.true_branch, Leaf) and isinstance(node.false_branch, Leaf)):
			return self.__trim_leaf(node, data, classes, verbose=verbose)
		else:
			true_split, false_split = node.condition.split(data)
			true_data = data.loc[true_split]
			true_classes = classes[true_split]

			false_data = data.loc[false_split]
			false_classes = classes[false_split]

			if(isinstance(node.true_branch, Node)):
				node.true_branch = self._post_pruning(node.true_branch, true_data, true_classes, verbose=verbose)

			if(isinstance(node.false_branch, Node)):
				node.false_branch = self._post_pruning(node.false_branch, false_data, false_classes, verbose=verbose)

			# One final trim 
			return self.__trim_leaf(node, data, classes, verbose=verbose)

	# A wrapper function for _post_pruning
	def prune(self, data, classes, verbose=0):
		self.root = self._post_pruning(self.root, data, classes, verbose=verbose)

if __name__ == '__main__':
	### Load data ###
	columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
				'marital-status', 'occupation', 'relationship', 'race',
				'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
				'native-country', 'class']

	data = pd.read_csv('adult.data', names=columns)

	features = data[list(data.columns)[:-1]]
	targets = data[list(data.columns)[-1]]

	### Build a decision tree ###
	clf = DecisionTreeClassifier(max_depth=10)

	### Train the decision tree ###
	X_train, X_test, Y_train, Y_test = train_test_split(features, targets, test_size=1/3)
	clf.fit(X_train, Y_train, verbose=1)

	### Prediction before pruning ###
	predictions = clf.predict(X_test)
	accuracy = accuracy_score(Y_test, predictions)
	print(accuracy)

	### Prune and make prediction after pruning ###
	clf.prune(X_test, Y_test, verbose=1)

	predictions = clf.predict(X_test)
	accuracy = accuracy_score(Y_test, predictions)
	print(accuracy)
