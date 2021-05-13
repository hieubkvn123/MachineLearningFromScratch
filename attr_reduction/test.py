import pandas as pd
from intuition_fuzzy import IntuitiveFuzzy

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
print(F.filter(verbose=True))

### Test Cases for sig and distance using sample2.csv ###
### --- Distance --- ###
# print(F.intuitive_partition_dist(['c1', 'c2'], ['c2', 'c3']))
# print(F.intuitive_partition_dist(['c2', 'c3'], ['c3', 'c4']))
# print(F.intuitive_partition_dist(['c1', 'c2'], ['c4', 'c3']))

### --- Sig --- ###
# print(F.sig([], 'c1'))
# print(F.sig([], 'c2'))
# print(F.sig([], 'c3'))
# print(F.sig([], 'c4'))
# print(F.sig([], 'c5'))
# print(F.sig([], 'c6'))
