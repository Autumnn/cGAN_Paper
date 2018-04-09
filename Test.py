import numpy as np


a = np.random.randint(10, size=(5,2))
print(a)
b = np.max(a, axis=0)
print(b)
print(np.max(a))
