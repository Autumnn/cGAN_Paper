import os
import numpy as np
from sklearn import svm, preprocessing, metrics
from sklearn.model_selection import StratifiedKFold
import Read_Data_UCI as RD
from imblearn.metrics import geometric_mean_score

#  first "min_max_scalar" ant then "StratifiedKFold".

path = "KEEL_npz"
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

    Features = np.concatenate((Positive_Features, Negative_Features))
    Labels = np.concatenate((Positive_Labels, Negative_Labels))

    Num_Cross_Folders = 5
    min_max_scalar = preprocessing.MinMaxScaler()
    Re_Features = min_max_scalar.fit_transform(Features)

    #skf = StratifiedKFold(n_splits=Num_Cross_Folders, shuffle=True)
    skf = StratifiedKFold(n_splits=Num_Cross_Folders, shuffle=False)
    G_Mean = np.linspace(0,0,Num_Cross_Folders)
    F_Mean = np.linspace(0,0,Num_Cross_Folders)
    AUC = np.linspace(0,0,Num_Cross_Folders)

    i = 0
    for train_index, test_index in skf.split(Re_Features, Labels):
        Feature_train_o, Feature_test = Re_Features[train_index], Re_Features[test_index]
        Label_train_o, Label_test = Labels[train_index], Labels[test_index]

#       print(np.array(np.nonzero(Label_train_o)).shape[1])
#       print(Label_train_o.shape[0])
#       print(np.array(np.nonzero(Label_test)).shape[1])
#       print(Label_test.shape[0])

        Num_Gamma = 12
        gamma = np.logspace(-2, 1, Num_Gamma)
        Num_C = 6
        C = np.logspace(-1, 4, Num_C)
        G_Mean_temp = np.zeros((Num_Gamma, Num_C))
        F_Mean_temp = np.zeros((Num_Gamma, Num_C))
        AUC_temp = np.zeros((Num_Gamma, Num_C))

        for j in range(Num_Gamma):
            for k in range(Num_C):
                #print("gamma = ", str(gamma[j]), " C = ", str(C[k]))
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

    file_wirte_AUC = "AUC_Result.txt"
    with open(file_wirte_AUC,'a') as w:
        AUC_line = name + '\t' + "SVM" + '\t'
        AUC_line += '\t'.join(str(x) for x in AUC)
        mean = np.mean(AUC)
        var = np.var(AUC)
        AUC_line += '\t' + str(mean) + '\t' + str(var) + '\n'
        w.write(AUC_line)

    file_wirte_G = "G_Result.txt"
    with open(file_wirte_G, 'a') as w_g:
        G_line = name + '\t' + "SVM" + '\t'
        G_line += '\t'.join(str(x) for x in G_Mean)
        mean = np.mean(G_Mean)
        var = np.var(G_Mean)
        G_line += '\t' + str(mean) + '\t' + str(var) + '\n'
        w_g.write(G_line)

    file_wirte_F = "F_Result.txt"
    with open(file_wirte_F, 'a') as w_f:
        F_line = name + '\t' + "SVM" + '\t'
        F_line += '\t'.join(str(x) for x in F_Mean)
        mean = np.mean(F_Mean)
        var = np.var(F_Mean)
        F_line += '\t' + str(mean) + '\t' + str(var) + '\n'
        w_f.write(F_line)