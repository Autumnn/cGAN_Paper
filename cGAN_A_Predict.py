from keras.layers import Input
from keras.models import load_model
from ipywidgets import IntProgress
import numpy as np
from numpy.linalg import cholesky
from sklearn import preprocessing
import matplotlib.pyplot as plt

File_Majority = "Generated_majority_samples.npy"
o_trans = np.load(File_Majority)
File_Minority = "Generated_minority_samples.npy"
s_trans = np.load(File_Minority)

num_majority = o_trans.shape[0]
num_minority = s_trans.shape[0]
#majority_labels = np.ones((num_majority,1))
#minority_labels = np.zeros((num_minority,1))
#condition_samples = np.concatenate((majority_labels, minority_labels))

input_dim = 20
print('Load Model')
G_dense = 160
D_dense = 80
Pre_train_epoches = 100
Train_epoches = 10000
Model_name = "cGAN_A_G-dense_" + str(G_dense) + "_pretrain_" + str(Pre_train_epoches) + "_D-dense_" + str(D_dense) + "_maintrain_" + str(Train_epoches) + ".h5"
model = load_model(Model_name)

print('Generate Samples')
Num_Create_samples = num_majority - num_minority
Noise_Input = np.random.uniform(0, 1, size=[Num_Create_samples, input_dim])
condition_samples = np.zeros((Num_Create_samples, 1))
Sudo_Samples = model.predict([Noise_Input, condition_samples])

plt.scatter(o_trans[:,0],o_trans[:,1], color = '#539caf', alpha = 0.3)
plt.scatter(s_trans[:,0], s_trans[:,1], marker = '+', color = 'r', label='2', s = 3)
#plt.scatter(condition_samples[:,0], condition_samples[:,1],marker = 'o', color = 'g', label='1', s = 3, alpha=1)
plt.scatter(Sudo_Samples[:,0], Sudo_Samples[:,1],marker = '^', color = 'rebeccapurple', label='1', s = 3, alpha=1)
plt.show()

