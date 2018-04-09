import numpy as np

a = np.random.randint(1,10,5)
print(a)
b = np.random.randint(11, 20, 3)
print(b)
a = np.concatenate((a, b))
print(a[4:10])