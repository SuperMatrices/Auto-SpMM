import matplotlib.pyplot as plt
import numpy as np
import seaborn
import pandas as pd
import csv
import numpy
from scipy import stats

def feature_loader(feature_file_name,
                   start_idx):
    feature_file = open(feature_file_name, "r")
    total_file_counter = 0

    feature_time = []
    features = []
    empty_feature_list = []
    
    line = feature_file.readline()
    line = feature_file.readline()
    while line:
        #print(line)
        line = line.strip()
        line = line.split(" ")
        if (len(line) == 1):  #too small to generate features
            #print(line)
            #print(total_file_counter)
            features.append([])
            feature_time.append(0)
            total_file_counter = total_file_counter + 1
            empty_feature_list.append(total_file_counter)
            line = feature_file.readline()  
            continue
        else:
            feature_time.append(float(line[-2]))
            line = feature_file.readline()
            line = line.strip()
            line = line.split(" ")
            tmp_feature = []
            for i in range(0, len(line)):
                tmp_feature.append(float(line[i]))
            features.append(tmp_feature)
            total_file_counter = total_file_counter + 1
            line = feature_file.readline()
            line = feature_file.readline()          
    
    return features, feature_time, empty_feature_list

def feature_csv_writer(train_data_file_name,
                       features):
    train_data_file = open(train_data_file_name, 'w', newline='')
    writer = csv.writer(train_data_file)
    
    writer.writerows(features)
    return

plt.rcParams['font.sans-serif'] = ['Times New Roman']
font1 = {'family' : 'Times New Roman',
         'weight' : 'bold',
         'size' : 20,
        }

#read data
k = [32, 64, 128]

mkl_files = ["./mkl_training"]
tc_files = ["./tc_training"]
cc_files = ["./cc_training"]

compute_tc_speedup = []
overall_tc_speedup = []
compute_cc_speedup = []
overall_cc_speedup = []
mkl_speedup = []

compute_tc_speedup2 = [[],[],[],[]]
overall_tc_speedup2 = [[],[],[],[]]
compute_cc_speedup2 = [[],[],[],[]]
overall_cc_speedup2 = [[],[],[],[]]
mkl_speedup2 = [[],[],[],[]]

for each in k:
    tmp_compute_tc_speedup = []
    tmp_compute_cc_speedup = []
    tmp_overall_tc_speedup = []
    tmp_overall_cc_speedup = []
    tmp_mkl_speedup = []
    for i in range(0, len(mkl_files)):
        mkl_file = open(mkl_files[i] + "_" + str(each) + ".log")
        tc_file = open(tc_files[i] + "_" + str(each) + ".log")
        cc_file = open(cc_files[i] + "_" + str(each) + ".log")
        
        #read mkl
        line = mkl_file.readline()
        line = mkl_file.readline()
        while line:
            line = line.strip()
            line = line.split(" ")
            tmp_mkl_speedup.append(float(line[-2]))
            line = mkl_file.readline()
            line = mkl_file.readline()

        line = cc_file.readline()
        line = cc_file.readline()
        while line:
            line = line.strip()
            line = line.split(" ")
            trans_time = float(line[-2])
            line = cc_file.readline()
            line = line.strip()
            line = line.split(" ")
            compute_time = float(line[-2])
            tmp_compute_cc_speedup.append(compute_time)
            tmp_overall_cc_speedup.append(trans_time + compute_time)
            line = cc_file.readline()
            line = cc_file.readline()

        line = tc_file.readline()
        line = tc_file.readline()
        line = tc_file.readline()
        line = tc_file.readline()
        while line:
            line = line.strip()
            line = line.split(" ")
            trans_time = float(line[-2])
            line = tc_file.readline()
            line = line.strip()
            line = line.split(" ")
            compute_time = float(line[-2])
            tmp_compute_tc_speedup.append(compute_time)
            tmp_overall_tc_speedup.append(trans_time + compute_time)
            line = tc_file.readline()
            line = tc_file.readline()
            line = tc_file.readline()
            line = tc_file.readline()
    mkl_speedup.append(tmp_mkl_speedup)
    compute_cc_speedup.append(tmp_compute_cc_speedup)
    compute_tc_speedup.append(tmp_compute_tc_speedup)
    overall_cc_speedup.append(tmp_overall_cc_speedup)
    overall_tc_speedup.append(tmp_overall_tc_speedup)

#generate speedup
for i in range(0, len(mkl_speedup)):
    for j in range(0, len(mkl_speedup[i])):
        base = mkl_speedup[i][j]
        compute_cc_speedup2[i].append(base / compute_cc_speedup[i][j])
        overall_cc_speedup2[i].append(base / overall_cc_speedup[i][j])
        compute_tc_speedup2[i].append(base / compute_tc_speedup[i][j])
        overall_tc_speedup2[i].append(base / overall_tc_speedup[i][j])
        mkl_speedup2[i].append(1)       

#load features
mat_feature_file_name = "../test/feature_test/feature_mtx.log"

mat_basic_features, mat_feature_times, mat_empty_feature_list = feature_loader(mat_feature_file_name, 0)

#check feature time
normalized_feature = []

for i in range(0, len(mkl_speedup)):
    for j in range(0, len(mkl_speedup[i])):
        normalized_feature.append(mat_feature_times[j] / mkl_speedup[i][j])
        
fig2= plt.figure(figsize=(8,3))
plt.hist(normalized_feature, bins=50, color='#3CB587', edgecolor='black', linewidth=0.1)
#plt.yscale('log')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Count", fontsize = 16, fontweight = 'bold')
plt.xlabel("# of MKL Iterations", fontsize = 16, fontweight = 'bold')
plt.grid(axis='x' ,linewidth=0.8, linestyle = '--', color='gray')
plt.savefig('overhead.pdf', dpi=120, bbox_inches='tight')
plt.show()



