from __future__ import print_function
import os
import numpy as np

def Initialize_Data(dir):
    Num_lines = len(open(dir, 'r').readlines())
    num_columns = 0
    data_info_lines = 0
    with open(dir, "r") as get_info:
        print("name", get_info.name)
        for line in get_info:
            if line.find("@") == 0:
                data_info_lines += 1
            else:
                columns = line.split(",")
                num_columns = len(columns)
                break

    global Num_Samples
    Num_Samples = Num_lines - data_info_lines
    print(Num_Samples)
    global Num_Features
    Num_Features = num_columns - 1

    global Features
    Features = np.ones((Num_Samples, Num_Features))
    global Labels
    Labels = np.ones((Num_Samples, 1))

    global Num_positive
    Num_positive = 0
    global Num_negative
    Num_negative = 0

    with open(dir, "r") as data_file:
        print("Read Data", data_file.name)
        l = 0
        for line in data_file:
            l += 1
            if l > data_info_lines:
                # print(line)
                row = line.split(",")
                length_row = len(row)
                # print('Row length',length_row)
                # print(row[0])
                #print(l)
                for i in range(length_row):
                    if i < length_row - 1:
                        if row[i] == '<null>':
                            row[i] = 0
                        Features[l - data_info_lines - 1][i] = row[i]
                        # print(Features[l-14][i])
                    else:
                        attri = row[i].strip()
                        # print(attri)
                        if attri == 'negative':
                            Labels[l - data_info_lines - 1][0] = 0
                            Num_negative += 1
                            # print(Labels[l-14][0])
                        else:
                            Labels[l - data_info_lines - 1][0] = 1
                            Num_positive += 1

#    print("Number of Positive: ", Num_positive)
    global Positive_Feature
    Positive_Feature = np.ones((Num_positive, Num_Features))
#    print("Num of Negative: ", Num_negative)
    global Negative_Feature
    Negative_Feature = np.ones((Num_negative, Num_Features))
    index_positive = 0
    index_negative = 0

    for i in range(Num_Samples):
        if Labels[i] == 1:
            Positive_Feature[index_positive] = Features[i]
            index_positive += 1
        else:
            Negative_Feature[index_negative] = Features[i]
            index_negative += 1

    print("Read Completed")

def get_feature():
    return Features

def get_label():
    return  Labels

def get_positive_feature():
    return Positive_Feature

def get_negative_feature():
    return Negative_Feature

path = "KEEL_Data"
files = os.listdir(path)
for file in files:
    print('File name: ', file)
    dir = path + '/' + file
    name = dir.split(".")[0].split("/")[1]
    Initialize_Data(dir)

    print(Positive_Feature[0])
    print(Positive_Feature.shape)
    print(Negative_Feature[0])
    print(Negative_Feature.shape)

    npy_name = name + ".npz"
    if Positive_Feature.shape[0] > Negative_Feature.shape[0]:
        np.savez(npy_name, P_F = Negative_Feature, N_F = Positive_Feature)
    else:
        np.savez(npy_name, P_F = Positive_Feature, N_F = Negative_Feature)
