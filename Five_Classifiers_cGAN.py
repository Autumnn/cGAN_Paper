import numpy as np
from keras.layers import Input
from keras.models import load_model
import cGAN as gan
from sklearn import svm, preprocessing, metrics
from sklearn.model_selection import StratifiedKFold
import Read_Data_UCI as RD
from imblearn.metrics import geometric_mean_score

#  first "min_max_scalar" ant then "StratifiedKFold".

dir = "UCI/yeast.data"
RD.Initialize_Data(dir)
name = dir.split(".")[0].split("/")[1]

Features = RD.get_feature()
Labels = RD.get_label().ravel()

Num_Cross_Folders = 5
min_max_scalar = preprocessing.MinMaxScaler()
Re_Features = min_max_scalar.fit_transform(Features)

input_dim = 20
G_dense = 160
D_dense = 80
print('Generate Models')
G_in = Input(shape=[input_dim])
C_in = Input(shape=[1])
G, G_out = gan.get_generative(G_in, C_in, dense_dim=G_dense, out_dim=RD.Num_Features)
G.summary()
D_in = Input(shape=[RD.Num_Features])
D, D_out = gan.get_discriminative(D_in, C_in, dense_dim=D_dense)
D.summary()
GAN_in = Input([input_dim])
GAN, GAN_out = gan.make_gan(GAN_in, C_in, G, D)
GAN.summary()

Pre_train_epoches = 100
Train_epoches = 10000
gan.pretrain(G, D, Labels, Re_Features, noise_dim=input_dim, epoches=Pre_train_epoches)
d_loss, g_loss = gan.train(GAN, G, D, Labels, Features, epochs=Train_epoches, noise_dim=input_dim, verbose=True)
Model_name = "cGAN_" + name + "_G-dense_" + str(G_dense) + "_pretrain_" + str(Pre_train_epoches) + "_D-dense_" + str(D_dense) + "_epoches_" + str(Train_epoches) + ".h5"
G.save(Model_name)

model = load_model(Model_name)
Num_Create_samples = RD.Num_negative - RD.Num_positive
Noise_Input = np.random.uniform(0, 1, size=[Num_Create_samples, input_dim])
condition_samples = np.linspace(1,1,Num_Create_samples)
print(Labels.shape)
print(condition_samples.shape)
Sudo_Samples = model.predict([Noise_Input, condition_samples])

#Re_Features = np.concatenate((Re_Features, Sudo_Samples))
#Labels = np.concatenate((Labels, condition_samples))

#skf = StratifiedKFold(n_splits=Num_Cross_Folders, shuffle=True)
skf = StratifiedKFold(n_splits=Num_Cross_Folders, shuffle=False)
G_Mean = np.linspace(0,0,Num_Cross_Folders)
F_Mean = np.linspace(0,0,Num_Cross_Folders)
AUC = np.linspace(0,0,Num_Cross_Folders)

i = 0
for train_index, test_index in skf.split(Re_Features, Labels):
    Feature_train_o, Feature_test = Re_Features[train_index], Re_Features[test_index]
    Label_train_o, Label_test = Labels[train_index], Labels[test_index]
    num = np.ceil(len(Sudo_Samples)/Num_Cross_Folders)
    Feature_train_o = np.concatenate((Feature_train_o, Sudo_Samples[int(i*num):int((i+1)*num)]))
    Label_train_o = np.concatenate((Label_train_o, condition_samples[int(i*num):int((i+1)*num)]))

    Num_Gamma = 12
    gamma = np.logspace(-2, 1, Num_Gamma)
    Num_C = 6
    C = np.logspace(-1, 4, Num_C)
    G_Mean_temp = np.zeros((Num_Gamma, Num_C))
    F_Mean_temp = np.zeros((Num_Gamma, Num_C))
    AUC_temp = np.zeros((Num_Gamma, Num_C))

    for j in range(Num_Gamma):
        for k in range(Num_C):
            print("gamma = ", str(gamma[j]), " C = ", str(C[k]))
            clf = svm.SVC(C=C[k], kernel='rbf', gamma=gamma[j])
            clf.fit(Feature_train_o, Label_train_o)
            Label_predict = clf.predict(Feature_test)

            G_Mean_temp[j, k] = geometric_mean_score(Label_test, Label_predict)
            F_Mean_temp[j, k] = metrics.f1_score(Label_test, Label_predict)
            Label_score = clf.decision_function(Feature_test)
            AUC_temp[j, k] = metrics.roc_auc_score(Label_test, Label_score)

#    print(G_Mean_temp)
#    print(F_Mean_temp)

    G_Mean[i] = np.max(G_Mean_temp)
    F_Mean[i] = np.max(F_Mean_temp)
    AUC[i] = np.max(AUC_temp)
    i += 1

file_wirte = "Result.txt"
with open(file_wirte,'a') as w:
    AUC_line = name + '\t' + "cGAN-SVM" + '\t' + "AUC" + '\t'
    AUC_line += '\t'.join(str(x) for x in AUC)
    mean = np.mean(AUC)
    var = np.var(AUC)
    AUC_line += '\t' + str(mean) + '\t' + str(var) + '\n'
    w.write(AUC_line)

    G_line = name + '\t' + "cGAN-SVM" + '\t' + "F_Mean" + '\t'
    G_line += '\t'.join(str(x) for x in G_Mean)
    mean = np.mean(G_Mean)
    var = np.var(G_Mean)
    G_line += '\t' + str(mean) + '\t' + str(var) + '\n'
    w.write(G_line)

    F_line = name + '\t' + "cGAN-SVM" + '\t' + "G_Mean" + '\t'
    F_line += '\t'.join(str(x) for x in F_Mean)
    mean = np.mean(F_Mean)
    var = np.var(F_Mean)
    F_line += '\t' + str(mean) + '\t' + str(var) + '\n'
    w.write(F_line)