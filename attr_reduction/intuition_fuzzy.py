""" Read Page 3 """
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

class IntuitiveFuzzy(object):
	def __init__(self, dataframe):
		super(IntuitiveFuzzy, self).__init__()
		assert isinstance(dataframe, pd.DataFrame)

		self.lambda_ = 1 

		print('[INFO] Initializing object ...')
		self.data = dataframe
		self.attributes = list(self.data.columns) # Including decision. Assume last column is decision values
		self.C = self.attributes[:-1]
		self.num_attr = len(self.attributes)
		self.num_objs = len(self.data.values)
		self.relational_matrices = self._get_single_attr_IFRM(self.data)
		print('[INFO] Done')


	def __str__(self):
		string = f"Attributes list : {str(self.attributes)}\n\n"
	
		for attr in self.attributes:
			string+=f'Relational matrix of {attr} : \n{str(self.relational_matrices[attr])}\n\n'

		return string

	def _get_single_attr_IFRM(self, data):
		"""
			This function returns a dictionary of relational matrices corresponding to
			each single attributes

			Params :
				- data : The pandas DataFrame of sample data 

			Returns : 
				- result : The dictionary of relational matrices corresponding each attribute 
		"""
		result = {}

		for attr in self.attributes:
			column = data[attr]
			rel_matrix = np.empty((self.num_objs, self.num_objs), dtype=tuple)

			for i in range(self.num_objs):
				for j in range(self.num_objs):
					mu = round(1 - abs(column[i] - column[j]), 2)
					v  = round((1 - mu) / (1 + self.lambda_ * mu), 2)

					rel_matrix[i][j] = (mu, v)

			result[attr] = rel_matrix

		return result

	def _intersect_ifr(self, tup_list):
		"""
			This function complements _get_multiple_attr_IFRM, it returns
			IFR(Q Union P) of the current tuple list 

			Params :
				- tup_list : A tuple list in the form (mu, v)

			Returns :
				- result : a tuple in the form of (inf(mu), sup(v))
		"""

		return (min(_x[0] for _x in tup_list), max(_x[1] for _x in tup_list))

	
	def _get_multiple_attr_IFRM(self, attributes):
		"""
			This function returns the intuitive relational matrix of two or more attributes

			Params :
				- attributes : List of attributes 

			Returns :
				- result : The relational matrix of the attributes partition
		"""
		assert len(attributes) >= 1
		if(len(attributes) == 1): return self.relational_matrices[attributes[0]]

		combined = np.empty((len(attributes), self.num_objs, self.num_objs), dtype=tuple)
		for i, attr in enumerate(attributes):
			combined[i,:,:] = self.relational_matrices[attr]

		combined = np.apply_along_axis(self._intersect_ifr, 0, combined)
		result = np.empty((self.num_objs, self.num_objs), dtype=tuple)

		for i in range(self.num_objs):
			for j in range(self.num_objs):
				result[i][j] = (combined[0, i, j], combined[1, i, j])

		return result

	def _get_intersect_IFRM(self, p1, p2):
		"""
			This function will return the intuitive  relational matrix of P intersect Q
			where P and Q are two sets of attributes which are both complete subset of C.
			Note : in the paper R{P intersect Q} = R{P} union R{Q}

			Params :
				- p1 : First subset of attributes
				- p2 : Second subset of attributes

			Returns :
				- result : The IFRM of P intersect Q
		"""
		result = np.empty((self.num_objs, self.num_objs), dtype=tuple)
		IFRM_1 = self._get_multiple_attr_IFRM(p1)
		IFRM_2 = self._get_multiple_attr_IFRM(p2)

		for i in range(self.num_objs):
			for j in range(self.num_objs):
				result[i, j] = (max(IFRM_1[i,j][0], IFRM_2[i,j][0]), min(IFRM_1[i, j][1], IFRM_2[i, j][1]))

		return result

	def _get_union_IFRM(self, p1, p2):
		"""
			This function will return the intuitive  relational matrix of P union Q
			where P and Q are two sets of attributes which are both complete subset of C.
			Note : in the paper R{P union Q} = R{P} intersect R{Q}

			Params :
				- p1 : First subset of attributes
				- p2 : Second subset of attributes

			Returns :
				- result : The IFRM of P intersect Q
		"""
		result = np.empty((self.num_objs, self.num_objs), dtype=tuple)
		IFRM_1 = self._get_multiple_attr_IFRM(p1)
		IFRM_2 = self._get_multiple_attr_IFRM(p2)

		for i in range(self.num_objs):
			for j in range(self.num_objs):
				result[i, j] = (min(IFRM_1[i,j][0], IFRM_2[i,j][0]), max(IFRM_1[i, j][1], IFRM_2[i, j][1]))

		return result

	def _is_intersected(self, p1, p2):
		""" 
			Check if two partitions of attributes intersect 
			
			Params :
				- p1 : First partition 
				- p2 : Second partition 

			Returns :
				- intersected : bool, whether the two paritions have intersection 
				- intersection : the intersection region of the two partition 
		"""
		intersected = False 
		intersection = []
		for attr in p1:
			if(attr in p2): 
				intersected = True 
				intersection.append(attr)

		return (intersected, intersection)

	def _get_cardinality(self, IFRM):
		"""
			Returns the caridnality of a partition of attributes 

			Params :
				- IFRM : An intuitive fuzzy relational matrix

			Returns :
				- caridnality : The caridnality of that parition 
		"""
		caridnality = 0
		for i in range(self.num_objs):
			for j in range(self.num_objs):
				mu = IFRM[i, j][0]
				v  = IFRM[i, j][1]

				caridnality += ((1 + mu - v) / 2)

		return caridnality

	def intuitive_partition_dist(self, p1, p2):
		"""
			This function returns the distance between two partitions of attributes. 
			Note : When calculating the distance between an empty partition and a non-empty
			partition. Just take the cardinality of the non-empty parition multiplied by |U|**2

			Params :
				- p1 : First partition of attributes 
				- p2 : Second partition of attributes 

			Returns :
				- result : A scalar representing the distance
		"""
		assert len(p1) > 0 or len(p2) > 0 # at least one of the partitions is non-empty

		### If one of the partitions is empty ###
		if(len(p1) == 0 or len(p2) == 0):
			if(len(p1) == 0):
				IFRM = self._get_multiple_attr_IFRM(p2)
				return round(self._get_cardinality(IFRM) * (1/(self.num_objs*self.num_objs)),2)

			elif(len(p2) == 0):
				IFRM = self._get_multiple_attr_IFRM(p1)
				return round(self._get_cardinality(IFRM) * (1/(self.num_objs*self.num_objs)),2)

		### If both partitions are non-empty ###
		intersected, intersection = self._is_intersected(p1, p2)

		union_IFRM = self._get_union_IFRM(p1, p2)
		inter_IFRM = self._get_intersect_IFRM(p1, p2)
	   
		union_cardinality = self._get_cardinality(union_IFRM)
		inter_cardinality = self._get_cardinality(inter_IFRM)

		return round((1 / ((self.num_objs)**2)) * (inter_cardinality - union_cardinality),2)

	def sig(self, B, a):
		"""
			This function measures the significance of an attribute a to the set of 
			attributes B

			Params :
				- B : list of attributes 
				- a : an attribute in C but not in B

			Returns :
				- sig : significance value of a to B
		"""
		assert isinstance(B, list)
		assert a not in B and a in self.C

		sig = 0

		d1 = self.intuitive_partition_dist(B, B + [self.attributes[-1]])
		d2 = self.intuitive_partition_dist(B + [a], B + [self.attributes[-1]] + [a])

		sig = d1 - d2

		return sig

	def filter(self, verbose=0):
		# initialization 
		B = []
		W = []
		d = self.intuitive_partition_dist(B, B + [self.attributes[-1]])
		D = self.intuitive_partition_dist(self.C, self.attributes)

		# Filter phase 
		while(d > D):
			max_sig = 0
			c_m = None
			for c in set(self.C).difference(set(B)):
				SIG_B_c = self.sig(B, c)
				if(SIG_B_c > max_sig):
					max_sig = SIG_B_c
					c_m = c 

			B.append(c_m)
			W.append(B.copy()) 

			# Re-calculate d
			d = self.intuitive_partition_dist(B, B + [self.attributes[-1]])

		return W

data_file = 'sample2.csv'
data = pd.read_csv(data_file, header=0)
# data = data[['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','a11','a12','a13','d']]
# data['d'] = LabelEncoder().fit_transform(data['d'].values)
# for i in list(data.columns[:-1]):
# 	values = data[i].values
# 	max_ = max(values)
# 	min_ = min(values)
# 	data[i] = (values - min_) / (max_ - min_)

F = IntuitiveFuzzy(data)
print(F.filter())

### Test Cases for sig and distance ###
### --- Distance --- ###
# print(F.intuitive_partition_dist(['a1', 'a2'], ['a2', 'a3']))
# print(F.intuitive_partition_dist(['a2', 'a3'], ['a3', 'a4']))
# print(F.intuitive_partition_dist(['a1', 'a2'], ['a4', 'a3']))

### --- Sig --- ###
# print(F.sig([], 'c1'))
# print(F.sig([], 'c2'))
# print(F.sig([], 'c3'))
# print(F.sig([], 'c4'))
# print(F.sig([], 'c5'))
# print(F.sig([], 'c6'))