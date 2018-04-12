import numpy as np

file = "AUC_Result.txt"
i = 0
with open(file, "r") as r:
    print(r.name)
    for line in r:
        column = line.split("\t")
        data_set = column[0]
        method = column[1]
        mean = float(column[7])
        var = float(column[8].strip())
        if i == 0 :
            dic = {data_set:{method:[mean, var]}}
            temp = data_set
        elif dic.__contains__(data_set):
            dic[data_set][method] = [mean, var]
        else:
            dic[data_set] = {method:[mean, var]}
        i += 1

method_list = dic[temp].keys()

file_write = "AUC_Result_Analysis_KEEL.txt"
with open(file_write, 'a') as w:
    first_line = "dataset" + '\t' + '\t'.join(str(x) + '\t' for x in method_list) + '\n'
    w.write(first_line)
    for key, values in dic.items():
        #print(values)
        l_m = []
        l_v = []
        for k, v in values.items():
            l_m.append(v[0])
            l_v.append(v[1])
        seq = sorted(l_m)
        w_line = key
        for i in range(len(l_m)):
            w_line += '\t' + str('%.4f' % l_m[i]) + '+' + str('%.4f' % l_v[i])
            w_line += '\t' + str(len(l_m)-seq.index(l_m[i]))
        w_line += '\n'
        w.write(w_line)



