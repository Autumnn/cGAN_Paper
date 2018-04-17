from __future__ import print_function
import os
import numpy as np
from sklearn import preprocessing

Features = np.array([[1,2,3,],[3,5,10],[7,8,10]])
print(Features)

min_max_scalar = preprocessing.MinMaxScaler()
Re_Features = min_max_scalar.fit_transform(Features)
print(Re_Features)