#auto-spmm model test
import numpy as np
import pandas as pd
import lightgbm
from lightgbm import log_evaluation, early_stopping, record_evaluation
import csv
import matplotlib.pyplot as plt
import seaborn as sns

#data transfer cost functions
def trans_cost_tc(feature, trans_ratio):
    tmp_M = feature[0]
    tmp_N = feature[1]
    tmp_K = feature[2]
    tmp_nnz = feature[3]
    tmp_A = tmp_nnz * 4 + tmp_M * 2
    tmp_B = tmp_N * tmp_K
    tmp_C = tmp_M * tmp_K
    total = tmp_A + tmp_B + tmp_C
    cost_A = (tmp_A) / total * (1 / trans_ratio)
    cost_B = (tmp_B) / total * (1 / trans_ratio)
    cost_C = (tmp_C) / total * (1 / trans_ratio)
    return cost_A, cost_B, cost_C

def trans_cost_cc(feature, trans_ratio):
    tmp_M = feature[0]
    tmp_N = feature[1]
    tmp_K = feature[2]
    tmp_nnz = feature[3]
    tmp_A = tmp_nnz * 2 + tmp_M
    tmp_B = tmp_N * tmp_K
    tmp_C = tmp_M * tmp_K
    total = tmp_A + tmp_B + tmp_C
    cost_A = (tmp_A) / total * (1 / trans_ratio)
    cost_B = (tmp_B) / total * (1 / trans_ratio)
    cost_C = (tmp_C) / total * (1 / trans_ratio)
    return cost_A, cost_B, cost_C

#tiled data computation cost functions
def tiled_compute_cost(model, feature, new_tile_size):
    new_features = []
    tmp_new_feature = []
    tmp_new_feature.append(new_tile_size)
    for each in feature:
        tmp_new_feature.append(each)
    new_features.append(tmp_new_feature)
    result = model.predict(new_features)
    return result

#pipeline acceleration evaluation
def pipeline_speedup(feature,
                     new_tile_size, 
                     tiled_compute_cost,
                     trans_A,
                     trans_B,
                     trans_C):
    tmp_dim = feature[2]
    ratio = new_tile_size / tmp_dim
    stage_num = (tmp_dim + new_tile_size - 1) // new_tile_size
    #print(stage_num)
    tmp_trans_B = trans_B * ratio
    tmp_trans_C = trans_C * ratio
    per_stage_cost = max(tmp_trans_B, tmp_trans_C, tiled_compute_cost)
    new_overall_cost = per_stage_cost * (stage_num - 1) + trans_A + tmp_trans_B + tmp_trans_C + tiled_compute_cost
    result = (trans_A + trans_B + trans_C + 1)/new_overall_cost
    return result


print("...Load Models...")

category3 = lightgbm.Booster(model_file='./saved_category3.txt')
category2 = lightgbm.Booster(model_file='./saved_category2.txt')
detector_cc_category = lightgbm.Booster(model_file='./gbdt_detector_cc_category.txt')
detector_tc_category = lightgbm.Booster(model_file='./gbdt_detector_tc_category.txt')
detector_cc_regression = lightgbm.Booster(model_file='./gbdt_detector_cc_regression.txt')
detector_tc_regression = lightgbm.Booster(model_file='./gbdt_detector_tc_regression.txt')

dim_cc_regression = lightgbm.Booster(model_file='./gbdt_dim_cc_regression.txt')
dim_tc_regression = lightgbm.Booster(model_file='./gbdt_dim_tc_regression.txt')

df_test = pd.read_csv('./category3_data.csv', header=None, sep=',')
y_base = df_test[0].values
X_test = df_test.drop(0,axis=1).values

#step-1: get preliminary prediction
y_pred = category3.predict(X_test, num_iteration=category3.best_iteration)

y_category3 = []
for each in y_pred:
    tmp_prediction = max(each)
    if (each[0] == tmp_prediction):
        y_category3.append(0)
    elif (each[1] == tmp_prediction):
        y_category3.append(1)
    else:
        y_category3.append(2)
print(y_category3)

#step-2: check pipeline potential
potential_flag = [] #1:potential, 0: not potential
y_category2 = []
y_pred = category2.predict(X_test, num_iteration=category2.best_iteration)
for each in y_pred:
    if (abs(1 - each) > abs(each)):
        y_category2.append(0)
    else:
        y_category2.append(1)

y_detector_cc_category = []
y_detector_cc_regress = []
y_detector_tc_category = []
y_detector_tc_regress = []
y_final_pred = []

y_pred = detector_cc_category.predict(X_test, num_iteration=detector_cc_category.best_iteration)
for each in y_pred:
    if (abs(1 - each) > abs(each)):
        y_detector_cc_category.append(0)
    else:
        y_detector_cc_category.append(1)

y_pred = detector_tc_category.predict(X_test, num_iteration=detector_cc_category.best_iteration)
for each in y_pred:
    if (abs(1 - each) > abs(each)):
        y_detector_tc_category.append(0)
    else:
        y_detector_tc_category.append(1)

y_pred = detector_cc_regression.predict(X_test, num_iteration=detector_cc_regression.best_iteration)
for each in y_pred:
    y_detector_cc_regress.append(pow(10, each))

y_pred = detector_tc_regression.predict(X_test, num_iteration=detector_tc_regression.best_iteration)

for each in y_pred:
    y_detector_tc_regress.append(pow(10, each))
print(y_detector_tc_regress)

min_tile_size = 32
for i in range(0, len(y_category3)):
    if (y_category3[i] == 2): #cpu
        if (y_category2[i] == 0): #cc
            if (y_detector_cc_category[i] == 0 and y_detector_cc_regress[i] > 0.75 and X_test[i][2] > min_tile_size):
                potential_flag.append(1)
                y_final_pred.append(-2)
            else:
                potential_flag.append(0)
                y_final_pred.append(2)
        else: #tc
            if (y_detector_tc_category[i] == 0 and y_detector_tc_regress[i] > 0.75 and X_test[i][2] > min_tile_size):
                potential_flag.append(1)
                y_final_pred.append(-1)
            else:
                potential_flag.append(0)
                y_final_pred.append(2)
    elif (y_category3[i] == 0): #cc
        if (y_detector_cc_category[i] == 0 and y_detector_cc_regress[i] > 0.5 and X_test[i][2] > min_tile_size):
            potential_flag.append(1)
        else:
            potential_flag.append(0)
        y_final_pred.append(0)
    else: #tc
        if (y_detector_tc_category[i] == 0 and y_detector_tc_regress[i] > 0.75 and X_test[i][2] > min_tile_size):
            potential_flag.append(1)
        else:
            potential_flag.append(0)
        y_final_pred.append(1)


#step-3: pipeline construction
pipeline_block_size = []

for i in range(0, len(potential_flag)):
    if (potential_flag[i] == 0): #unpotential
        pipeline_block_size.append(-1)
    else:
        #predict cost of each matrices
        if (y_final_pred[i] == 1 or y_final_pred[i] == -1): #tc
            overall_trans = y_detector_tc_regress[i]
            cost_A, cost_B, cost_C = trans_cost_tc(X_test[i], overall_trans)
            tile_size = min_tile_size
            speedups = []
            while tile_size < X_test[i][2]:
                tmp_compute_cost = tiled_compute_cost(dim_tc_regression, X_test[i], tile_size)
                speedups.append(pipeline_speedup(X_test[i], 
                                                 tile_size,
                                                 tmp_compute_cost,
                                                 cost_A,
                                                 cost_B,
                                                 cost_C))
                tile_size = tile_size * 2
            print(y_category3[i])
            if (max(speedups) > 1.25):
                pipeline_block_size.append(min_tile_size * pow(2, speedups.index(max(speedups))))
            else:
                pipeline_block_size.append(-1)
        else: #cc
            overall_trans = y_detector_cc_regress[i]
            cost_A, cost_B, cost_C = trans_cost_cc(X_test[i], overall_trans)
            tile_size = min_tile_size
            speedups = []
            while tile_size < X_test[i][2]:
                tmp_compute_cost = tiled_compute_cost(dim_cc_regression, X_test[i], tile_size)
                speedups.append(pipeline_speedup(X_test[i], 
                                                 tile_size,
                                                 tmp_compute_cost,
                                                 cost_A,
                                                 cost_B,
                                                 cost_C))
                tile_size = tile_size * 2
            if (max(speedups) > 1.25):
                pipeline_block_size.append(min_tile_size * pow(2, speedups.index(max(speedups))))
            else:
                pipeline_block_size.append(-1)


#speedup compute

#read data
k = [32, 64, 128, 256]

mkl_files = ["./mkl_training"]
tc_files = ["./tc_training"]
cc_files = ["./cc_training"]
cusparse_files = ["./cusparse_training_mtx_3090"]

compute_tc_speedup = []
overall_tc_speedup = []
compute_cc_speedup = []
overall_cc_speedup = []
cusparse_speedup = []
mkl_speedup = []

compute_tc_speedup2 = [[],[],[],[]]
overall_tc_speedup2 = [[],[],[],[]]
compute_cc_speedup2 = [[],[],[],[]]
overall_cc_speedup2 = [[],[],[],[]]
cusparse_speedup2 = [[],[],[],[]]
mkl_speedup2 = [[],[],[],[]]

for each in k:
    tmp_overall_tc_speedup = []
    tmp_overall_cc_speedup = []
    tmp_mkl_speedup = []
    tmp_cusparse_speedup = []
    for i in range(0, len(mkl_files)):
        mkl_file = open(mkl_files[i] + "_" + str(each) + ".log")
        tc_file = open(tc_files[i] + "_" + str(each) + ".log")
        cc_file = open(cc_files[i] + "_" + str(each) + ".log")
        cusparse_file = open(cusparse_files[i] + "_" + str(each) + ".log")
        
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
            tmp_overall_tc_speedup.append(trans_time + compute_time)
            line = tc_file.readline()
            line = tc_file.readline()
            line = tc_file.readline()
            line = tc_file.readline()

        line = cusparse_file.readline()
        line = cusparse_file.readline()
        while line:
            line = line.strip()
            line = line.split(" ")
            overall_time = float(line[-2])
            tmp_cusparse_speedup.append(overall_time)
            line = cusparse_file.readline()
            line = cusparse_file.readline()

    mkl_speedup.append(tmp_mkl_speedup)
    overall_cc_speedup.append(tmp_overall_cc_speedup)
    overall_tc_speedup.append(tmp_overall_tc_speedup)
    cusparse_speedup.append(tmp_cusparse_speedup)


sample_counter = 0
pipeline_speedups = []
ccs = []
tcs = []
mkls = []
autos = []
cusparse = []
gpu_only = []
print(len(mkl_speedup[0]))
for i in range(0, len(mkl_speedup)):
    for j in range(0, len(mkl_speedup[i])):
        ccs.append(mkl_speedup[i][j]/overall_cc_speedup[i][j])
        tcs.append(mkl_speedup[i][j]/overall_tc_speedup[i][j])
        cusparse.append(mkl_speedup[i][j]/(cusparse_speedup[i][j] + overall_cc_speedup[i][j] - compute_cc_speedup[i][j]))
        mkls.append(1)

for i in range(0, len(pipeline_block_size)):
    if (pipeline_block_size[i] == -1):
        if (y_category3[i] == 0):
            autos.append(ccs[i])
        elif (y_category3[i] == 1):
            autos.append(tcs[i])
        else:
            autos.append(mkls[i])
    else:
        autos.append(pipeline_speedups[sample_counter])
        sample_counter = sample_counter + 1

print(sum(autos) / len(autos))
print(sum(ccs) / len(ccs))
toccs = 0
toccs2 = []
totcs = 0
totcs2 = []
tocusparse = 0
tocusparse2 = []
for i in range(0, len(autos)):
    toccs = toccs + autos[i] / ccs[i]  
    toccs2.append(autos[i] / ccs[i])
    totcs = totcs + autos[i] / tcs[i]  
    totcs2.append(autos[i] / tcs[i] )
    tocusparse = tocusparse + autos[i]/cusparse[i]
    tocusparse2.append(autos[i]/cusparse[i])

print(toccs / len(autos), totcs/ len(autos), sum(autos)/len(autos), tocusparse/len(autos))

#generate dataframe
final_data = []
for i in range(0, len(ccs)):
    tmp_tuple = ('GE-SpMM', int(X_test[i][2]), ccs[i])
    final_data.append(tmp_tuple)

for i in range(0, len(tcs)):
    tmp_tuple = ('TC-SpMM', int(X_test[i][2]), tcs[i])
    final_data.append(tmp_tuple)
    
for i in range(0, len(cusparse)):
    tmp_tuple = ('cuSPARSE', int(X_test[i][2]), cusparse[i])
    final_data.append(tmp_tuple)

for i in range(0, len(autos)):
    tmp_tuple = ('Auto-SpMM', int(X_test[i][2]), autos[i])
    final_data.append(tmp_tuple)

df_final = pd.DataFrame.from_records(final_data, columns=['libraries', 'K', 'speedup'])

fig = plt.figure(figsize=(8,4))
plt.rcParams['font.sans-serif'] = ['Times New Roman']

plt.yscale('log')

medianprops = {'linestyle': '-', 'color': 'black'}
mat_ids = range(0, len(autos))

ax = sns.boxplot(data=df_final, x="K", y="speedup", hue="libraries", showmeans=True, meanprops={'marker': '^', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': 6}, palette=['#A7C877', '#AED1FE', '#EEDC75', '#8BA5B6'], fliersize=0.5) 

plt.legend(bbox_to_anchor=(0.5, 1.1), ncol=4, loc = 'upper center', fontsize=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Normalized Speedup to MKL", fontsize = 16, fontweight = 'bold')
plt.xlabel("K", fontsize = 16, fontweight = 'bold')
plt.savefig('result.pdf', dpi=120, bbox_inches='tight')
plt.show()



