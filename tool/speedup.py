# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import matplotlib

stencils_2d = ["2d9pt_star", "2d81pt_box", "2d121pt_box", "2d5pt_arbitrary_shape", "2d5pt_nested"]
stencils_3d = ["3d13pt_star", "3d27pt_box", "3d125pt_box", "3d7pt_arbitrary_shape", "3d7pt9pt_nested"]
stencil_names = stencils_2d + stencils_3d

# 20 个迭代步的时间(ms)
iterations = 20
dsl_2d_times = [459.899, 2109.804, 3127.004, 454.814, 2772.589]
dsl_3d_times = [930.266, 737.201, 3228.068, 642.329, 6625.212]
athread_2d_times = [460.256, 2110.053, 3126.84, 451.498, 2781.963]
athread_3d_times = [936.822, 757.524, 3228.103, 641.504, 6470.641]
openacc_2d_times = [953.169, 2303.64, 3374.724, 730.202, 2800.439]
openacc_3d_times = [1448.522, 895.48, 4050.358, 790.823, 7565.621]
msc_2d_times = [507.4, 2193.71, 3745, 499.16, 0]
msc_3d_times = [984.2, 753.14, 5202.59, 979.01, 0]

# tune 优化后的时间
# 只有2d5pt_nested和3d7pt9pt_nested
dsl_fusion_2d_times = [459.899, 2109.804, 3127.004, 454.814, 1321.145]
dsl_fusion_3d_times = [930.266, 737.201, 3228.068, 642.329, 5068.118]
# fusion之后进行向量化优化
dsl_vector_2d_times = [459.899, 1030.023, 1513.762, 454.814, 1321.145]
dsl_vector_3d_times = [930.266, 737.201, 2871.896, 642.329, 5068.118]

dsl_fusion_times = dsl_fusion_2d_times + dsl_fusion_3d_times
dsl_vector_times = dsl_vector_2d_times + dsl_vector_3d_times

dsl_times = dsl_2d_times + dsl_3d_times
athread_times = athread_2d_times + athread_3d_times
openacc_times = openacc_2d_times + openacc_3d_times
msc_times = msc_2d_times + msc_3d_times

# 以openacc为基准, 计算加速比
dsl_speedup = []
dsl_total_speedup = []
dsl_vector_speedup = []
dsl_fusion_speedup = []
dsl_fusion_speedup_real = []
athread_speedup = []
openacc_speedup = [1 for _ in range(len(stencil_names))]
msc_speedup = []

for i in range(len(stencil_names)):
    dsl_speedup.append(openacc_times[i]/dsl_times[i]) # fusion base
    dsl_fusion_speedup_real.append(openacc_times[i]/dsl_fusion_times[i]) # vector base
    dsl_fusion_speedup.append(openacc_times[i]/dsl_fusion_times[i] - openacc_times[i]/dsl_times[i])
    dsl_vector_speedup.append(openacc_times[i]/dsl_vector_times[i] - openacc_times[i]/dsl_fusion_times[i])
    dsl_total_speedup.append(openacc_times[i]/dsl_vector_times[i])
    athread_speedup.append(openacc_times[i]/athread_times[i])
    if (msc_times[i] == 0):
        msc_speedup.append(0)
    else:
        msc_speedup.append(openacc_times[i]/msc_times[i])

# 开始绘图
plt.rc('font', family='Times New Roman')
plt.rcParams['figure.figsize'] = (10.0, 3)
fig, ax = plt.subplots()

offset = [0.055, 0.165]
lala = np.arange(0, 5, 0.5)
width = 0.1

ax.bar(lala-offset[1], openacc_speedup, width=width, label="openACC(f64)", edgecolor='Black', linewidth=0.7, color='#C0C0C0', hatch='---')
ax.bar(lala-offset[0], athread_speedup, width=width, label="Athread(f64)", edgecolor='Black', linewidth=0.7, color='#C0C0C0', hatch='..')
ax.bar(lala+offset[0], msc_speedup, width=width, label="MSC(f64)", edgecolor='Black', linewidth=0.7, color='#C0C0C0', hatch='\\\\')
ax.bar(lala+offset[1], dsl_speedup, width=width, label="SWStenDSL(f64)", edgecolor='Black', linewidth=0.7, color='#FFDEAD', hatch='//')
ax.bar(lala+offset[1], dsl_fusion_speedup, bottom = dsl_speedup, width=width, label="SWStenDSL(f64) fusion", edgecolor='Black', linewidth=0.7, color='#3C5182', hatch='x')
ax.bar(lala+offset[1], dsl_vector_speedup, bottom = dsl_fusion_speedup_real, width=width, label="SWStenDSL(f64) vector", edgecolor='Black', linewidth=0.7, color='#EFA59A', hatch='+')

yticks = [0.5, 1.0, 1.5, 2, 2.5]
ax.set_yticks(yticks)
ax.set_ylabel('Speedup',fontsize=16)
ax.set_xticks(lala)
ax.set_xticklabels(stencil_names, fontsize=12)
for tick in ax.get_xticklabels():
    tick.set_rotation(20)

for i in range(len(lala)):
    ax.text(lala[i]-offset[0], athread_speedup[i], '%.2fx' % athread_speedup[i], ha='center', va='bottom', fontsize=6)
    if (msc_speedup[i] != 0):
        ax.text(lala[i]+offset[0], msc_speedup[i]+0.1, '%.2fx' % msc_speedup[i], ha='center', va='bottom', fontsize=6)
    ax.text(lala[i]+offset[1], dsl_total_speedup[i]+0.1, '%.2fx' % dsl_total_speedup[i], ha='center', va='bottom', fontsize=6)

ax.legend(loc='upper right',fontsize=8,ncol=2)
plt.tight_layout()
plt.show()

fig.savefig('speedup.pdf')