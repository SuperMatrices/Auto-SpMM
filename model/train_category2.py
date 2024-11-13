import matplotlib.pyplot as plt
import numpy as np
import seaborn
import pandas as pd
import csv
import numpy
from scipy import stats
import random
import lightgbm as lgb 
from lightgbm import log_evaluation, early_stopping, record_evaluation
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

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

#generate category2 features
total_file_counter = 0
category2_features = []
for i in range(0, len(mkl_speedup)):
    for j in range(0, len(mkl_speedup[i])):
        cc = overall_cc_speedup[i][j]
        tc = overall_tc_speedup[i][j]                      
        tmp_feature = []
        if (cc < tc):
            tmp_feature.append(0)
        else:
            tmp_feature.append(1)
        for each in mat_basic_features[j]:
            tmp_feature.append(each)
        tmp_feature[3] = k[i]
        category2_features.append(tmp_feature)

feature_csv_writer("./category2_data.csv", category2_features)

#split the csv file
#k-fold cross validation
train_data = open("./category2_data.csv", 'r')
data_reader = csv.reader(train_data)
data_writers = []
write_file_name = []

for i in range(0, 10):
    tmp_file = open("./category2_data_" + str(i) + ".csv", 'w', newline='')
    tmp_data_writer = csv.writer(tmp_file)
    write_file_name.append(tmp_file)
    data_writers.append(tmp_data_writer)

for row in data_reader:
    tmp_tag = row[0]
    if (row[0] == '0'):
        tmp_seed = random.randint(0, 9)
        data_writers[tmp_seed].writerow(row)
    else:
        tmp_seed = random.randint(0, 9)
        data_writers[tmp_seed].writerow(row)

for i in range(0, 10):
    write_file_name[i].close()

#model-1 category2 prediction

print("...Load Data...")

k_fold_seed = 9
accuracy = []
precision = []
recall = []
f1 = []
class_num = 3

for k_fold_seed in range(0,10):
    fold_data_name = []
    train_data_list = []

    for i in range(0, 10):
        if (i != k_fold_seed):
            tmp_file_name = "./category2_data_" + str(i) + ".csv"
            fold_data_name.append(tmp_file_name)
            tmp_reader = pd.read_csv(tmp_file_name, header=None, sep=',')
            train_data_list.append(tmp_reader)

    df_train = pd.concat(train_data_list, axis=0)
    df_test = pd.read_csv('./category2_data_' + str(k_fold_seed) + ".csv", header=None, sep=',')

    y_train = df_train[0].values
    y_test = df_test[0].values
    X_train = df_train.drop(0, axis=1).values
    print(X_train)
    X_test = df_test.drop(0, axis=1).values
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'num_leaves': 31,
        'metric': {'l2', 'auc'},
        'is_unbalance':True,
        'max_bin' : 2047,
        'learning_rate': 0.08,  
        'feature_fraction': 0.95,
        'bagging': 0.7,
        'bagging_freq': 5,
        'lambda_l1':0.05, 
        'lambda_l2':0.2,
        'num_threads': 16,
        'verbose': 1    
    }
    #set the early stopping and log period
    evals_result = {}
    callbacks = [log_evaluation(period=10), early_stopping(stopping_rounds=15), record_evaluation(evals_result)]

    gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=(lgb_eval, lgb_train),
                valid_names=('validate','train'),
                #evals_result=evals_result,
                callbacks=callbacks)
      
    predictions = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    
    classification_result = []
    for each in predictions:
        if (abs(1 - each) > abs(each)):
            classification_result.append(0)
        else:
            classification_result.append(1)

    
gbm.save_model('saved_category2.txt')