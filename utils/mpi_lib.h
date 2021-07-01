/**
 * @file mpi_lib.h
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief 本头文件定义了大规模并行计算使用的通信库
 * @version 0.1
 * @date 2021-06-27
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#ifndef _SWSTEN_MPI_LIB_H_
#define _SWSTEN_MPI_LIB_H_

#include <mpi.h>

// 声明2D通信模板函数, 该函数需要完成8个方向的信息收发操作
// 定义的exchange_halo_2D_TYPE函数传参的含义
// input: 进行数据交换的数组
// dim1, dim0: input数组最高维和最低维的大小
// mpi_dim1, mpi_dim0: mpi进程划分情况, 最高维度和最低维度的大小
// halo_*: 四个方向halo的大小
// 相对位置关系, 及tag分配
// wn | n | en   0 | 1 | 2 
// ---|---|---  ---|---|---
//  w | c | e    3 |   | 4
// ---|---|---  ---|---|---
// ws | s | es   5 | 6 | 7
// s, e方向为索引值增长方向
#define EXCHANGE_HALO_2D_DECLARE(TYPE)\
void exchange_halo_2D_##TYPE(TYPE *input, int dim1, int dim0, int mpi_dim1, int mpi_dim0,\
            int halo_n, int halo_s, int halo_w, int halo_e, int pid);

// 声明3D通信模板函数, 该函数需要完成26个方向的信息收发操作
// 定义的exchange_halo_3D_TYPE函数传参的含义
// input: 进行数据交换的数组
// dim2, dim1, dim: input数组最高维到最低维的大小
// halo_*: 六个方向的halo大小
// 相对位置关系, 及tag分配
// 1. 最底面:
// bwn|bn |ben   0 | 1 | 2 
// ---|---|---  ---|---|---
// bw |bc |be    3 | 4 | 5
// ---|---|---  ---|---|---
// bws|bs |bes   6 | 7 | 8
// 2. 中间面
// mwn|mn |men  10 |11 |12 
// ---|---|---  ---|---|---
// mw |mc |me   13 |14 |15
// ---|---|---  ---|---|---
// mws|ms |mes  16 |17 |18
// 3. 最顶面
// twn|tn |ten  20 |21 |22 
// ---|---|---  ---|---|---
// tw |tc |te   23 |24 |25
// ---|---|---  ---|---|---
// tws|ts |tes  26 |27 |28
// s, e, b 方向为索引值增长方向
#define EXCHANGE_HALO_3D_DECLARE(TYPE)\
void exchange_halo_3D_##TYPE(TYPE *input, int dim2, int dim1, int dim0,\
                        int mpi_dim2, int mpi_dim1, int mpi_dim0,\
                        int halo_t, int halo_b,\
                        int halo_n, int halo_s,\
                        int halo_w, int halo_e,  int pid);

// 获取本从核在通信域中的rank
int mpiGetMyRank();

EXCHANGE_HALO_2D_DECLARE(float)
EXCHANGE_HALO_2D_DECLARE(double)

EXCHANGE_HALO_3D_DECLARE(float)
EXCHANGE_HALO_3D_DECLARE(double)
#endif /* endof _SWSTEN_MPI_LIB_H_ */
