#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

if len(sys.argv) != 2 :
    print('Usage: ', sys.argv[0], 'FileName')

is_in_spe_module = False
is_in_mpe_module = False
is_in_share = False
is_move_into_region_has_value = False
is_in_move_into_region = False
output_string = ''
share_list = ''
spe_func_declare = ''
spe_func_move_into_region = ''
spe_func_counter = 0
spe_kernel_counter = 0
output_file=None
discrete_var = 0    # 这个变量用于将主核, 从核使用的变量名称做区分, 
                    # 使变量的形式为value$(discrete_var)_[0-9]+,
                    # 从而避免从核__thread_local变量重定义问题

# 如果MPE源文件存在则删除该文件, 因为该文将将以追加的形式打开
mpe_src_path = "kernel"+sys.argv[1]+"_master.c"
if os.path.exists(mpe_src_path) is True:
    os.remove(mpe_src_path)

#输出文件
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    for line in f.readlines():
        # 判断当前行是否有$命令
        output_string = ''
        if (line.find("$") != -1):
            # 存在，则说明该行需要执行相应$命令
            if (line.find("moduleBegin") != -1): # spe_module开始命令
                is_in_spe_module = True
                output_file = open("kernel"+sys.argv[1]+str(spe_kernel_counter)+"_slave.c", 'w')
            elif (line.find("moduleEnd") != -1): # spe_module结束命令
                is_in_spe_module = False
                spe_kernel_counter = spe_kernel_counter + 1
                discrete_var = discrete_var + 1
                output_file.close()
            elif (line.find("mainModuleBegin") != -1): # mpe_module开始命令
                is_in_mpe_module = True
                output_file = open("kernel"+sys.argv[1]+"_master.c", 'a+')
            elif (line.find("mainModuleEnd") != -1): # mpe_module结束命令
                is_in_mpe_module = False
                discrete_var = discrete_var + 1
                output_file.close()
            elif (line.find("shareBegin") != -1): # share域开始命令
                is_in_share = True
            elif (line.find("shareEnd") != -1): # share域结束命令
                is_in_share = False
            elif (line.find("speDeclare") != -1): # spe_func声明命令
                command_index = line.find("speDeclare")
                spe_func_declare += line[0:command_index-1] + "\n"
            elif (line.find("moveInToRegionBegin") != -1):  # spe_func_moveInToRegion命令
                spe_func_move_into_region = ''
                is_in_move_into_region = True
            elif (line.find("moveInToRegionEnd") != -1):  # spe_func_moveInToRegionEnd命令
                is_move_into_region_has_value = True
                is_in_move_into_region = False
            elif (line.find("moveToHead") != -1): # moveToHead命令
                # 获取命令中的内容
                command_index = line.find("moveToHead<-")
                suffix = line[0:command_index-1].strip()
                prefix = line[command_index+12:-1]
                # 输出
                output_string = prefix + " " + suffix + "\n"
            elif (line.find("shareInsertPoint") != -1): # 共享域插入位置命令
                output_string = share_list + spe_func_declare
                share_list = ''
                spe_func_declare = ''
            elif (line.find("error") != -1):
                print("Find error!!! please reference to the %s", sys.argv[1])
            # $delete无需任何操作, 直接忽略
        elif (is_in_spe_module):
            # 该行不存在$命令, 但是在spe_module中
            if (is_in_share):
                # 当前行在share域内
                share_list += line
                output_string = line
            elif (is_in_move_into_region):
                # 当前行位于moveInToRegion域中
                spe_func_move_into_region += line
            else:
                output_string = line
            
            # 判断moveIntoRegion命令是否写入
            if (is_move_into_region_has_value) :
                output_string += spe_func_move_into_region
                is_move_into_region_has_value = False
        elif (is_in_mpe_module):
            # 该行不存在$命令, 但是在main_module中
            output_string = line
        output_string = output_string.replace("%", "value"+str(discrete_var)+"_")
        print(output_string, end='')
        if (len(output_string) != 0):
            output_file.write(output_string)
    
f.close()
