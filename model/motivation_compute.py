#motivation
import matplotlib.pyplot as plt
import numpy as np
import seaborn

plt.rcParams['font.sans-serif'] = ['Times New Roman']
font1 = {'family' : 'Times New Roman',
         'weight' : 'bold',
         'size' : 20,
        }

#read data
k = [32, 64, 128, 256]

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

#plot figure
x = range(len(mkl_speedup2[0]))

colors = ['grey', '#598A18', '#FFA500']
fig = plt.figure(figsize=(18,3))

ax1 = fig.add_subplot(1,3,1)
ax1.plot(x, mkl_speedup2[0], color=colors[0], label="MKL", linewidth=0.5)
ax1.plot(x, compute_cc_speedup2[0], color=colors[2], label="GE-SpMM", linewidth=0.5)
ax1.plot(x, compute_tc_speedup2[0], color=colors[1], label="TC-SpMM", linewidth=0.5)
ax1.set_ylim(0.01, 1000)
ax1.set_yscale('log')
ax1.set_title("(a) K = 32", y=-0.3, fontweight='bold', fontsize = 16)

ax2 = fig.add_subplot(1,3,2)
ax2.plot(x, mkl_speedup2[1], color=colors[0], label="MKL", linewidth=0.5)
ax2.plot(x, compute_cc_speedup2[1], color=colors[2], label="GE-SpMM", linewidth=0.5)
ax2.plot(x, compute_tc_speedup2[1], color=colors[1], label="TC-SpMM", linewidth=0.5)
ax2.set_ylim(0.01, 1000)
ax2.set_yscale('log')
ax2.set_title("(b) K = 64", y=-0.3, fontweight='bold', fontsize = 16)

ax3 = fig.add_subplot(1,3,3)
ax3.plot(x, mkl_speedup2[0], color=colors[0], label="MKL", linewidth=0.5)
ax3.plot(x, compute_cc_speedup2[2], color=colors[2], label="GE-SpMM", linewidth=0.5)
ax3.plot(x, compute_tc_speedup2[2], color=colors[1], label="TC-SpMM", linewidth=0.5)
ax3.set_ylim(0.01, 1000)
ax3.set_yscale('log')
ax3.set_title("(c) K = 128", y=-0.3, fontweight='bold', fontsize = 16)

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, bbox_to_anchor=(0.5, 1.05), ncol=3, loc = 'upper center', fontsize=14)

ax1.tick_params(axis='y', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)
ax3.tick_params(axis='y', labelsize=14)
ax1.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='x', labelsize=14)
ax3.tick_params(axis='x', labelsize=14)

ax1.set_ylabel('Speedup', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('compute.pdf', dpi=120, bbox_inches='tight')


