import numpy as np
from keras.layers import Input
from keras.models import load_model
import cGAN as gan
from sklearn import svm, preprocessing, metrics
from sklearn.model_selection import StratifiedKFold
import os
import cGANStructure
from imblearn.metrics import geometric_mean_score
from sklearn.datasets import make_gaussian_quantiles
from Reidjohnson.smote import SMOTEBoost


path = "UCI_Test"
files= os.listdir(path) #Get files in the folder
for file in files:
    print("File Name: ", file)
    name = file.split(".")[0]
    dir = path + "/" + file
    r = np.load(dir)

    Positive_Features = r["P_F"]
    Num_Positive = Positive_Features.shape[0]
    Positive_Labels = np.linspace(1,1,Num_Positive)
    Negative_Features = r["N_F"]
    Num_Negative = Negative_Features.shape[0]
    Negative_Labels = np.linspace(0,0,Num_Negative)
    Num_Features = Positive_Features.shape[1]

    Features = np.concatenate((Positive_Features, Negative_Features))
    Labels = np.concatenate((Positive_Labels, Negative_Labels))

    Num_Cross_Folders = 5
    min_max_scalar = preprocessing.MinMaxScaler()
    Re_Features = min_max_scalar.fit_transform(Features)

    skf = StratifiedKFold(n_splits=Num_Cross_Folders, shuffle=False)
    G_Mean = np.linspace(0, 0, Num_Cross_Folders)
    F_Mean = np.linspace(0, 0, Num_Cross_Folders)
    AUC = np.linspace(0, 0, Num_Cross_Folders)

    i = 0
    for train_index, test_index in skf.split(Re_Features, Labels):
        Feature_train_o, Feature_test = Re_Features[train_index], Re_Features[test_index]
        Label_train_o, Label_test = Labels[train_index], Labels[test_index]
        num_positive = np.array(np.nonzero(Label_train_o)).shape[1]
        print(num_positive)
        num_negative = Label_train_o.shape[0] - num_positive
        print(num_negative)
        Num_Create_samples= num_negative - num_positive

        smboost = SMOTEBoost(n_samples=Num_Create_samples, n_estimators=100)
        smboost.fit(Feature_train_o, Label_train_o)

        Label_predict = smboost.predict(Feature_test)

        G_Mean[i] = geometric_mean_score(Label_test, Label_predict)
        F_Mean[i] = metrics.f1_score(Label_test, Label_predict)
        Label_score = smboost.decision_function(Feature_test)
        AUC[i] = metrics.roc_auc_score(Label_test, Label_score)
        i += 1

    print(G_Mean)
    print(F_Mean)
    print(AUC)