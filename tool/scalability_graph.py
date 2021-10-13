# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

stencils_2d = ["2d9pt_star", "2d81pt_box", "2d121pt_box", "2d5pt_arbitrary_shape", "2d5pt_nested"]
stencils_3d = ["3d13pt_star", "3d27pt_box", "3d125pt_box", "3d7pt_arbitrary_shape", "3d7pt9pt_nested"]
stencil_names = stencils_2d + stencils_3d

num_ops = [17, 161, 241, 9, 18, 25, 53, 249, 13, 30]

sunway_weak_shape_grid_2d=[[16,8], [16,16], [32,16], [32,32]]
sunway_weak_shape_sub_grid_2d = [[4096,4096], [4096,4096], [4096,4096], [4096,4096]]
sunway_weak_shape_grid_3d=[[8,4,4], [8,4,8], [8,8,8], [16,8,8]]
sunway_weak_shape_sub_grid_3d = [[256,256,256], [256,256,256], [256,256,256], [256,256,256]]

sunway_strong_shape_grid_2d=[[16,8], [16,16], [32,16], [32,32]]
sunway_strong_shape_sub_grid_2d = [[4096,4096], [4096,2048], [2048,2048], [2048,1024]]
sunway_strong_shape_grid_3d=[[8,4,4], [8,4,8], [8,8,8], [16,8,8]]
sunway_strong_shape_sub_grid_3d = [[256,256,256], [256,256,128], [256,128,128], [128,128,128]]

# 20个迭代的时间(ms)
iterations = 20
sunway_weak_times = [[567.135 ,568.705, 572.073, 579.111],
                        [2269.094, 2270.109, 2283.602, 2281.243],
                        [3305.812, 3306.691, 3305.883, 3313.962],
                        [506.913, 508.1, 509.924, 514.165],
                        [3065.162, 3074.837, 3083.854, 3085.706],
                        [1848.622, 1863.522, 1903.958, 1890.405],
                        [1452.558, 1461.584, 1481.954, 1487.3],
                        [4210.32, 4221.152, 4259.894, 4265.896],
                        [1086.637, 1093.097, 1105.348, 1110.638],
                        [8190.945, 8267.383, 8269.596, 8290.919]]

sunway_strong_times = [[568.182, 335.998, 173.266, 112.68],
                          [2267.659, 1191.217, 599.855, 336.42],
                          [3301.499, 1702.073, 872.066, 471.713],
                          [508.978, 276.506, 142.473, 87.67],
                          [3071.062, 1583.823, 816.875, 441.716],
                          [1845.246, 1169.553, 660.879, 380.7],
                          [1451.274, 911.177, 507.814, 390.265],
                          [4198.575, 2325.458, 1243.309, 671.761],
                          [1082.127, 658.932, 364.375, 210.969],
                          [8200.284, 4439.073, 2312.224, 1242.985]]

def mul_list(list1):
    res=1
    for ele in list1:
        res=res*ele
    return res

def list_div_list(list1, list2):
    list3=[]
    for i in range(len(list1)):
        list3.append(list1[i]/list2[i])
    return list3

def list_div_num(list1, num):
    list2=[]
    for ele in list1:
        list2.append(ele/num)
    return list2

def lists_div_num(list1, num):
    list2=[]
    for row in list1:
        list2.append(list_div_num(row, num))
    return list2

def get_size_grid(shape_grid, shape_sub_grid):
    list1 = []
    for i in range(len(shape_grid)):
        list1.append(mul_list(shape_grid[i]) * mul_list(shape_sub_grid[i]))
    return list1

def get_flop(list1, num_op):
    list2=[]
    for ele in list1:
        list2.append(ele*num_op)
    return list2

def get_flops(list1, list2):
    list3=[]
    for i in range(len(list1)):
        list3.append(list_div_list(list1[i], list2[i]))
    return list3

def get_idea_flops(flops, num_mpi):
    base=[]
    for row in flops:
        base.append(row[0])
    res=[]
    for ele in base:
        list1=[]
        for i in range(num_mpi):
            list1.append(ele)
            ele=ele*2
        res.append(list1)
    return res

sunway_weak_flop = []
for i in range(len(stencils_2d)+len(stencils_3d)):
    if i < len(stencils_2d):
        sunway_weak_flop.append(get_flop(get_size_grid(sunway_weak_shape_grid_2d, sunway_weak_shape_sub_grid_2d), num_ops[i]))
    else:
        sunway_weak_flop.append(get_flop(get_size_grid(sunway_weak_shape_grid_3d, sunway_weak_shape_sub_grid_3d), num_ops[i]))

sunway_weak_flops = get_flops(sunway_weak_flop, lists_div_num(sunway_weak_times, 20*1000))
sunway_weak_Gflops = lists_div_num(sunway_weak_flops, 1000000000)
idea_sunway_weak_Gflops = get_idea_flops(sunway_weak_Gflops, 4)
print(sunway_weak_Gflops)

sunway_strong_flop = []
for i in range(len(stencils_2d)+len(stencils_3d)):
    if i < len(stencils_2d):
        sunway_strong_flop.append(get_flop(get_size_grid(sunway_strong_shape_grid_2d, sunway_strong_shape_sub_grid_2d), num_ops[i]))
    else:
        sunway_strong_flop.append(get_flop(get_size_grid(sunway_strong_shape_grid_3d, sunway_strong_shape_sub_grid_3d), num_ops[i]))
    
sunway_strong_flops = get_flops(sunway_strong_flop, lists_div_num(sunway_strong_times, 20*1000))
sunway_strong_Gflops = lists_div_num(sunway_strong_flops, 1000000000)
idea_sunway_strong_Gflops = get_idea_flops(sunway_strong_Gflops, 4)

weak_yvalue_list = [100, 400, 1600, 6400, 25600]
weak_ylabel_list = ['100','400','1.6K','6.4K','25.6K']
strong_yvalue_list = [100, 400, 1600, 6400, 25600]
strong_ylabel_list = ['100', '400', '1.6K', '6.4K', '25.6K']

def log_list(list1, logn=10):
    list2=[]
    for ele in list1:
        list2.append(math.log(ele, logn))
    return list2

def log_lists(list1, logn=10):
    list2=[]
    for row in list1:
        list2.append(log_list(row, logn))
    return list2
logn = 10
weak_log_yvalue_list = log_list(weak_yvalue_list,logn)
strong_log_yvalue_list = log_list(strong_yvalue_list,logn)
log_sunway_weak_Gflops = log_lists(sunway_weak_Gflops,logn)
log_sunway_strong_Gflops = log_lists(sunway_strong_Gflops,logn)
log_idea_sunway_weak_Gflops = log_lists(idea_sunway_weak_Gflops,logn)
log_idea_sunway_strong_Gflops = log_lists(idea_sunway_strong_Gflops,logn)

print(log_sunway_weak_Gflops)

plt.rc('font',family='Times New Roman')
plt.rcParams['figure.figsize'] = (20.5, 4.5)
nrow=2
ncol=10
fig, ax = plt.subplots(nrow,ncol)

def draw_picture():
    data_fs = 12
    fs = 10
    fs_ylabel=12

    # weak
    row = 0
    l_idea = ''
    l_real = ''
    for i in range(len(stencil_names)):
        col = i
        xindex = np.arange(len(log_sunway_weak_Gflops[i]))
        l_idea, = ax[row][col].plot(xindex, log_idea_sunway_weak_Gflops[i], linestyle = '--')
        l_real, = ax[row][col].plot(xindex, log_sunway_weak_Gflops[i], alpha = 0.7, marker='o', markersize=4)
        ax[row][col].set_xticks([])
        ax[row][col].set_yticks(weak_log_yvalue_list)
        ax[row][col].set_yticklabels([])
    ax[row][0].set_yticklabels(weak_ylabel_list, fontsize = fs)
    ax[row][0].set_ylabel('GFlops (Log Scale)',fontsize=fs_ylabel)
    ax[row][len(stencil_names)-1].legend((l_idea, l_real), ('SunWay-Ideal', 'Sunay-Real'), fontsize = 8, loc = 'upper left')

    # strong
    row = 1
    xlabel_list = ['128', '256', '512', '1024']
    for i in range(len(stencil_names)):
        col = i
        xindex = np.arange(len(log_sunway_strong_Gflops[i]))
        ax[row][col].plot(xindex, log_idea_sunway_strong_Gflops[i], linestyle = '--')
        ax[row][col].plot(xindex, log_sunway_strong_Gflops[i], alpha = 0.7, marker='o', markersize=4)

        ax[row][col].set_xlabel(stencil_names[i],fontsize=data_fs)
        ax[row][col].set_xticks([0,1,2,3])
        ax[row][col].set_xticklabels(xlabel_list, fontsize=fs)
        ax[row][col].set_yticklabels([])
        for tick in ax[row][col].get_xticklabels():
            tick.set_rotation(40)
        ax[row][col].set_yticks(strong_log_yvalue_list)
        # ax[row][col].set_yticklabels(strong_ylabel_list, fontsize=fs)
    ax[row][0].set_yticklabels(strong_ylabel_list, fontsize=fs)
    ax[row][0].set_ylabel('GFlops (Log Scale)',fontsize=fs_ylabel)

    fig.subplots_adjust(hspace = 0.2)
    fig.text(0.5, 0.56, "(a) Weak Scalability", horizontalalignment='center', fontsize=12)
    fig.text(0.5, 0.01, "(b)  Strong Scalability ", horizontalalignment='center', fontsize=12)

def main():
    draw_picture()
    plt.tight_layout()
    plt.show()

    fig.savefig('scala.pdf')

if __name__ == '__main__':
    main()