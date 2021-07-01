/**
 * @file mpi_lib.c
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief 本文件定义了大规模通信库的函数实现
 * @version 0.1
 * @date 2021-07-10
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "mpi_lib.h"

/**************************** 二维问题域halo区交换 *******************************/
// 获取当前进程在mpi网格中的位置
void pid_to_index_2D(const int pid, const int mpi_dim0, int *index_1, int *index_0)
{
    *index_1 = pid / mpi_dim0;
    *index_0 = pid % mpi_dim0;
}

// 判断指定位置的邻居是否存在(是否在mpi的网格内)
int within_bound_2D(const int index1, const int index0, const int mpi_dim1, const int mpi_dim0)
{
    if (index1<0 || index1>=mpi_dim1 || index0<0 || index0>=mpi_dim0)
        return 0;
    return 1;
}

// 根据位置计算pid
int index_to_pid_2D(int *pid, const int mpi_dim0, const int index1, const int index0) {
    *pid = index1*mpi_dim0 + index0;
}

// 模板函数
// 在同一方向上(同行, 同列, 同对角线)交换数据, 分别为邻居a和邻居b
// 向邻居a发送, 从邻居b接收
#define EXCHANGE_DATA_2D(TYPE, MPI_TYPE)\
void exchange_data_2D_##TYPE(TYPE *input, const int tag, const int dim0,\
                        const int a_index1, const int a_index0,\
                        const int b_index1, const int b_index0,\
                        const int mpi_dim1, const int mpi_dim0,\
                        const int data_width, const int data_height,\
                        const int send_for_start, const int send_for_end, const int send_for_offset,\
                        const int recv_for_start, const int recv_for_end, const int recv_for_offset)\
{\
    MPI_Status status_s, status_r;\
    MPI_Request handler_s, handler_r;\
    TYPE *buffer_send = NULL;\
    TYPE *buffer_recv = NULL;\
    /* 标志位, 表明是否进行了发送和接收操作 */\
    int flag_send = 0;\
    int flag_recv = 0;\
    /********************** 向邻居a发送, 从邻居b接收 **************************/\
    if (within_bound_2D(a_index1, a_index0, mpi_dim1, mpi_dim0)) {\
        /* 如果邻居a存在, 则向邻居a发送数据 */\
        int dest_pid;\
        index_to_pid_2D(&dest_pid, mpi_dim0, a_index1, a_index0);\
        flag_send = 1;\
        buffer_send = (TYPE *)malloc(data_width*data_height*sizeof(TYPE));\
        /* 数据打包 */\
        int cnt = 0;\
        int i;\
        for (i = send_for_start; i < send_for_end; i++) {\
            memcpy(&buffer_send[cnt], &input[i*dim0+send_for_offset], data_width*sizeof(TYPE));\
            cnt += data_width;\
        }\
        /* 数据发送 */\
        MPI_Isend(buffer_send, data_width*data_height, MPI_TYPE, dest_pid, tag, MPI_COMM_WORLD, &handler_s);\
    }\
    if (within_bound_2D(b_index1, b_index0, mpi_dim1, mpi_dim0)) {\
        /* 如果邻居b存在则从邻居b处接收数据 */\
        int dest_pid;\
        index_to_pid_2D(&dest_pid, mpi_dim0, b_index1, b_index0);\
        flag_recv = 1;\
        buffer_recv = (TYPE *)malloc(data_width*data_height*sizeof(TYPE));\
        MPI_Irecv(buffer_recv, data_width*data_height, MPI_TYPE, dest_pid, tag, MPI_COMM_WORLD, &handler_r);\
    }\
    /* 等待发送接收完成 */\
    if (flag_send) {\
        MPI_Wait(&handler_s, &status_s);\
        free(buffer_send);\
        buffer_send = NULL;\
    }\
    if (flag_recv) {\
        MPI_Wait(&handler_r, &status_r);\
        /* 数据解包 */\
        int cnt = 0;\
        int i;\
        for (i = recv_for_start; i < recv_for_end; i++) {\
            memcpy(&input[i*dim0+recv_for_offset], &buffer_recv[cnt], data_width*sizeof(TYPE));\
            cnt += data_width;\
        }\
        free(buffer_recv);\
        buffer_recv = NULL;\
    }\
}

// 定义2D通信模板函数, 该函数需要完成8个方向的信息收发操作
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
#define EXCHANGE_HALO_2D(TYPE)\
void exchange_halo_2D_##TYPE(TYPE *input, int dim1, int dim0, int mpi_dim1, int mpi_dim0,\
            int halo_n, int halo_s, int halo_w, int halo_e, int pid)\
{\
    /* 计算当前pid在整个网格中的位置 */\
    int index1, index0;\
    pid_to_index_2D(pid, mpi_dim0, &index1, &index0);\
    /* 开始交换halo区域 */\
    int width = dim0-halo_w-halo_e; /* 非halo区域宽度 */\
    int height = dim1-halo_n-halo_s; /* 非halo区域高度 */\
    /* 1. 向邻居n发送, 从邻居s接收, tag = 1 */\
    if (dim0*halo_s != 0) {\
        exchange_data_2D_##TYPE(input, /*tag=*/1, dim0,\
                        /*a_index1=*/index1-1, /*a_index0=*/index0,\
                        /*b_index1=*/index1+1, /*b_index0=*/index0,\
                        mpi_dim1, mpi_dim0,\
                        /*data_width=*/dim0, /*data_height=*/halo_s,\
                        /*sendfor_start=*/halo_n, /*sendfor_end=*/halo_n+halo_s, /*sendfor_offset*/0,\
                        /*recvfor_start=*/dim1-halo_s, /*recvfor_end=*/dim1, /*recvfor_offset*/0);\
    }\
    /* 2. 向邻居s发送, 从邻居n接收, tag = 6 */\
    if (dim0*halo_n != 0) {\
        exchange_data_2D_##TYPE(input, /*tag=*/6, dim0,\
                        /*a_index1=*/index1+1, /*a_index0=*/index0,\
                        /*b_index1=*/index1-1, /*b_index0=*/index0,\
                        mpi_dim1, mpi_dim0,\
                        /*data_width=*/dim0, /*data_height=*/halo_n,\
                        /*sendfor_start=*/height, /*sendfor_end=*/dim1-halo_s, /*sendfor_offset=*/0,\
                        /*recvfor_start=*/0, /*recvfor_end=*/halo_n, /*recvfor_offset=*/0);\
    }\
    /* 3. 向邻居w发送, 从邻居e接收, tag = 3 */\
    if (dim1*halo_e != 0) {\
        exchange_data_2D_##TYPE(input, /*tag=*/3, dim0,\
                        /*a_index1=*/index1, /*a_index0=*/index0-1,\
                        /*b_index1=*/index1, /*b_index0=*/index0+1,\
                        mpi_dim1, mpi_dim0,\
                        /*data_width=*/halo_e, /*data_height=*/dim1,\
                        /*sendfor_start=*/0, /*sendfor_end=*/dim1, /*sendfor_offset=*/halo_w,\
                        /*recvfor_start=*/0, /*recvfor_end=*/dim1, /*recvfor_offset=*/halo_w+width);\
    }\
    /* 4. 向邻居e发送, 从邻居w接收, tag = 4 */\
    if (dim1*halo_w != 0) {\
        exchange_data_2D_##TYPE(input, /*tag=*/4, dim0,\
                        /*a_index1=*/index1, /*a_index0=*/index0+1,\
                        /*b_index1=*/index1, /*b_index0=*/index0-1,\
                        mpi_dim1, mpi_dim0,\
                        /*data_width=*/halo_w, /*data_height=*/dim1,\
                        /*sendfor_start=*/0, /*sendfor_end=*/dim1, /*sendfor_offset=*/width,\
                        /*recvfor_start=*/0, /*recvfor_end=*/dim1, /*recvfor_offset=*/0);\
    }\
    /* 5. 向邻居wn发送, 从邻居es接收, tag = 0 */\
    if (halo_s*halo_e != 0) {\
        exchange_data_2D_##TYPE(input, /*tag=*/0, dim0,\
                        /*a_index1=*/index1-1, /*a_index0=*/index0-1,\
                        /*b_index1=*/index1+1, /*b_index0=*/index0+1,\
                        mpi_dim1, mpi_dim0,\
                        /*data_width=*/halo_e, /*data_height=*/halo_s,\
                        /*sendfor_start=*/halo_n, /*sendfor_end=*/halo_n+halo_s, /*sendfor_offset=*/halo_w,\
                        /*recvfor_start=*/halo_n+height, /*recvfor_end=*/dim1, /*recvfor_offset=*/halo_w+width);\
    }\
    /* 6. 向邻居es发送, 从邻居wn接收, tag = 7 */\
    if (halo_n*halo_w != 0) {\
        exchange_data_2D_##TYPE(input, /*tag=*/7, dim0,\
                        /*a_index1=*/index1+1, /*a_index0=*/index0+1,\
                        /*b_index1=*/index1-1, /*b_index0=*/index0-1,\
                        mpi_dim1, mpi_dim0,\
                        /*data_width=*/halo_w, /*data_height=*/halo_n,\
                        /*sendfor_start=*/height, /*sendfor_end=*/halo_n+height, /*sendfor_offset=*/width,\
                        /*recvfor_start=*/0, /*recvfor_end=*/halo_n, /*recvfor_offset=*/0);\
    }\
    /* 7. 向邻居en发送, 从邻居ws接收, tag = 2 */\
    if (halo_s*halo_w != 0) {\
        exchange_data_2D_##TYPE(input, /*tag=*/2, dim0,\
                        /*a_index1=*/index1-1, /*a_index0=*/index0+1,\
                        /*b_index1=*/index1+1, /*b_index0=*/index0-1,\
                        mpi_dim1, mpi_dim0,\
                        /*data_width=*/halo_w, /*data_height=*/halo_s,\
                        /*sendfor_start=*/halo_n, /*sendfor_end=*/halo_n+halo_s, /*sendfor_offset=*/width,\
                        /*recvfor_start=*/halo_n+height, /*recvfor_end=*/dim1, /*recvfor_offset=*/0);\
    }\
    /* 8. 向邻居ws发送, 从邻居en接收, tag = 5 */\
    if (halo_n*halo_e != 0) {\
        exchange_data_2D_##TYPE(input, /*tag=*/5, dim0,\
                        /*a_index1=*/index1+1, /*a_index0=*/index0-1,\
                        /*b_index1=*/index1-1, /*b_index0=*/index0+1,\
                        mpi_dim1, mpi_dim0,\
                        /*data_width=*/halo_e, /*data_height=*/halo_n,\
                        /*sendfor_start=*/height, /*sendfor_end=*/halo_n+height, /*sendfor_offset=*/halo_w,\
                        /*recvfor_start=*/0, /*recvfor_end=*/halo_n, /*recvfor_offset=*/width+halo_w);\
    }\
}

/*********************** 三维问题域halo区交换 ************************************/
// 获取当前进程在mpi网格中的位置
void pid_to_index_3D(const int pid, const int mpi_dim1, const int mpi_dim0,\
                    int *index_2, int *index_1, int *index_0)
{
    *index_2 = pid / (mpi_dim1*mpi_dim0);
    *index_1 = (pid % (mpi_dim1*mpi_dim0)) / mpi_dim0;
    *index_0  = pid % mpi_dim0;
}

// 判断指定位置的邻居是否存在(是否在mpi网格内)
int within_bound_3D(const int index2, const int index1, const int index0,\
                    const int mpi_dim2, const int mpi_dim1, const int mpi_dim0)
{
    if (index2<0 || index2 >= mpi_dim2 || index1<0 || index1 >= mpi_dim1\
        || index0<0 || index0 >= mpi_dim0)
        return 0;
    return 1;
}

// 根据位置计算pid
int index_to_pid_3D(int *pid, const int mpi_dim1, const int mpi_dim0,\
                const int index2, const int index1, const int index0)
{
    *pid = index2*mpi_dim1*mpi_dim0 + index1*mpi_dim0 + index0;
}

// 模板函数
// 在同一方向上(同x, 同y, 同z, 同对角线)交换数据, 分别为邻居a和邻居b
// 向邻居a发送, 从邻居b接收
#define EXCHANGE_DATA_3D(TYPE, MPI_TYPE)\
void exchange_data_3D_##TYPE(TYPE *input, const int tag, const int dim1, const int dim0,\
                        const int a_index2, const int a_index1, const int a_index0,\
                        const int b_index2, const int b_index1, const int b_index0,\
                        const int mpi_dim2, const int mpi_dim1, const int mpi_dim0,\
                        const int data_dim2, const int data_dim1, const int data_dim0,\
                        const int send_for_dim2_start, const int send_for_dim2_end,\
                        const int send_for_dim1_start, const int send_for_dim1_end,\
                        const int send_for_dim0_offset,\
                        const int recv_for_dim2_start, const int recv_for_dim2_end,\
                        const int recv_for_dim1_start, const int recv_for_dim1_end,\
                        const int recv_for_dim0_offset)\
{\
    MPI_Status status_s, status_r;\
    MPI_Request handler_s, handler_r;\
    TYPE *buffer_send = NULL;\
    TYPE *buffer_recv = NULL;\
    /* 标志位, 表明是否进行了发送和接收操作 */\
    int flag_send = 0;\
    int flag_recv = 0;\
    /************************* 向邻居a发送, 从邻居b接收 **************************/\
    if (within_bound_3D(a_index2, a_index1, a_index0, mpi_dim2, mpi_dim1, mpi_dim0)) {\
        /* 如果邻居a存在, 则向邻居a发送数据 */\
        int dest_pid;\
        index_to_pid_3D(&dest_pid, mpi_dim1, mpi_dim0, a_index2, a_index1, a_index0);\
        flag_send = 1;\
        buffer_send = (TYPE *)malloc(data_dim2*data_dim1*data_dim0*sizeof(TYPE));\
        /* 数据打包 */\
        int cnt = 0;\
        int i, j;\
        for (i = send_for_dim2_start; i < send_for_dim2_end; i++) {\
            for (j = send_for_dim1_start; j < send_for_dim1_end; j++) {\
                memcpy(&buffer_send[cnt], &input[i*dim1*dim0 + j*dim0 + send_for_dim0_offset], data_dim0*sizeof(TYPE));\
                cnt += data_dim0;\
            }\
        }\
        /* 数据发送 */\
        MPI_Isend(buffer_send, data_dim2*data_dim1*data_dim0, MPI_TYPE, dest_pid, tag, MPI_COMM_WORLD, &handler_s);\
    }\
    if (within_bound_3D(b_index2, b_index1, b_index0, mpi_dim2, mpi_dim1, mpi_dim0)) {\
        /* 如果邻居b存在, 则从邻居b处接收数据 */\
        int dest_pid;\
        index_to_pid_3D(&dest_pid, mpi_dim1, mpi_dim0, b_index2, b_index1, b_index0);\
        flag_recv = 1;\
        buffer_recv = (TYPE *)malloc(data_dim2*data_dim1*data_dim0*sizeof(TYPE));\
        MPI_Irecv(buffer_recv, data_dim2*data_dim1*data_dim0, MPI_TYPE, dest_pid, tag, MPI_COMM_WORLD, &handler_r);\
    }\
    /* 等待发送接收完成 */\
    if (flag_send) {\
        MPI_Wait(&handler_s, &status_s);\
        free(buffer_send);\
        buffer_send = NULL;\
    }\
    if (flag_recv) {\
        MPI_Wait(&handler_r, &status_r);\
        /* 数据解包 */\
        int cnt = 0;\
        int i, j;\
        for (i = recv_for_dim2_start; i<recv_for_dim2_end; i++) {\
            for (j = recv_for_dim1_start; j < recv_for_dim1_end; j++) {\
                memcpy(&input[i*dim1*dim0 + j*dim0 + recv_for_dim0_offset], &buffer_recv[cnt], data_dim0*sizeof(TYPE));\
                cnt += data_dim0;\
            }\
        }\
        free(buffer_recv);\
        buffer_recv = NULL;\
    }\
}

// 定义3D通信模板函数, 该函数需要完成26个方向的信息收发操作
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
#define EXCHANGE_HALO_3D(TYPE)\
void exchange_halo_3D_##TYPE(TYPE *input, int dim2, int dim1, int dim0,\
                        int mpi_dim2, int mpi_dim1, int mpi_dim0,\
                        int halo_t, int halo_b,\
                        int halo_n, int halo_s,\
                        int halo_w, int halo_e,  int pid)\
{\
    /* 计算当前pid在整个网格中的位置 */\
    int index2, index1, index0;\
    pid_to_index_3D(pid, mpi_dim1, mpi_dim0, &index2, &index1, &index0);\
    /* 开始交换halo区 */\
    int core_dim2 = dim2-halo_t-halo_b; /* 非halo区最高维大小 */\
    int core_dim1 = dim1-halo_n-halo_s; /* 非halo区中间维大小 */\
    int core_dim0 = dim0-halo_w-halo_e; /* 非halo区最低维大小 */\
    /* 1. 向邻居mn发送, 从邻居ms接收, tag=11 */\
    if (dim2*halo_s*dim0 != 0) {\
        exchange_data_3D_##TYPE(input, 11, dim1, dim0,\
                        /*a_index2=*/index2, /*a_index1=*/index1-1, /*a_index0=*/index0,\
                        /*b_index2=*/index2, /*b_index1=*/index1+1, /*b_index0=*/index0,\
                        mpi_dim2, mpi_dim1, mpi_dim0,\
                        /*data_dim2=*/dim2, /*data_dim1=*/halo_s, /*data_dim0=*/dim0,\
                        /*send_for_dim2_start=*/0, /*send_for_dim2_end=*/dim2,\
                        /*send_for_dim1_start=*/halo_n, /*send_for_dim1_end=*/halo_n+halo_s,\
                        /*send_for_dim_offset=*/0,\
                        /*recv_for_dim2_start=*/0, /*recv_for_dim2_end=*/dim2,\
                        /*recv_for_dim1_start=*/dim1-halo_s, /*recv_for_dim1_end=*/dim1,\
                        /*recv_for_dim0_offset=*/0);\
    }\
    /* 2. 向邻居ms发送, 从邻居mn接收, tag=17 */\
    if (dim2*halo_n*dim0 != 0) {\
        exchange_data_3D_##TYPE(input, 17, dim1, dim0,\
                        /*a_index2=*/index2, /*a_index1=*/index1+1, /*a_index0=*/index0,\
                        /*b_index2=*/index2, /*b_index1=*/index1-1, /*b_index0=*/index0,\
                        mpi_dim2, mpi_dim1, mpi_dim0,\
                        /*data_dim2=*/dim2, /*data_dim1=*/halo_n, /*data_dim0=*/dim0,\
                        /*send_for_dim2_start=*/0, /*send_for_dim2_end=*/dim2,\
                        /*send_for_dim1_start=*/core_dim1, /*send_for_dim1_end=*/dim1-halo_s,\
                        /*send_for_dim_offset=*/0,\
                        /*recv_for_dim2_start=*/0, /*recv_for_dim2_end=*/dim2,\
                        /*recv_for_dim1_start=*/0, /*recv_for_dim1_end=*/halo_n,\
                        /*recv_for_dim0_offset=*/0);\
    }\
    /* 3. 向邻居mw发送, 从邻居me接收, tag= 13*/\
    if (dim2*dim1*halo_e != 0) {\
        exchange_data_3D_##TYPE(input, 13, dim1, dim0,\
                        /*a_index2=*/index2, /*a_index1=*/index1, /*a_index0=*/index0-1,\
                        /*b_index2=*/index2, /*b_index1=*/index1, /*b_index0=*/index0+1,\
                        mpi_dim2, mpi_dim1, mpi_dim0,\
                        /*data_dim2=*/dim2, /*data_dim1=*/dim1, /*data_dim0=*/halo_e,\
                        /*send_for_dim2_start=*/0, /*send_for_dim2_end=*/dim2,\
                        /*send_for_dim1_start=*/0, /*send_for_dim1_end=*/dim1,\
                        /*send_for_dim_offset=*/halo_w,\
                        /*recv_for_dim2_start=*/0, /*recv_for_dim2_end=*/dim2,\
                        /*recv_for_dim1_start=*/0, /*recv_for_dim1_end=*/dim1,\
                        /*recv_for_dim0_offset=*/halo_w+core_dim0);\
    }\
    /* 4. 向邻居me发送, 从邻居mw接收, tag=15 */\
    if (dim2*dim1*halo_w != 0) {\
        exchange_data_3D_##TYPE(input, 15, dim1, dim0,\
                        /*a_index2=*/index2, /*a_index1=*/index1, /*a_index0=*/index0+1,\
                        /*b_index2=*/index2, /*b_index1=*/index1, /*b_index0=*/index0-1,\
                        mpi_dim2, mpi_dim1, mpi_dim0,\
                        /*data_dim2=*/dim2, /*data_dim1=*/dim1, /*data_dim0=*/halo_w,\
                        /*send_for_dim2_start=*/0, /*send_for_dim2_end=*/dim2,\
                        /*send_for_dim1_start=*/0, /*send_for_dim1_end=*/dim1,\
                        /*send_for_dim_offset=*/core_dim0,\
                        /*recv_for_dim2_start=*/0, /*recv_for_dim2_end=*/dim2,\
                        /*recv_for_dim1_start=*/0, /*recv_for_dim1_end=*/dim1,\
                        /*recv_for_dim0_offset=*/0);\
    }\
    /* 5. 向邻居mwn发送, 从邻居mes接收, tag = 10 */\
    if (dim2*halo_s*halo_e != 0) {\
        exchange_data_3D_##TYPE(input, 10, dim1, dim0,\
                        /*a_index2=*/index2, /*a_index1=*/index1-1, /*a_index0=*/index0-1,\
                        /*b_index2=*/index2, /*b_index1=*/index1+1, /*b_index0=*/index0+1,\
                        mpi_dim2, mpi_dim1, mpi_dim0,\
                        /*data_dim2=*/dim2, /*data_dim1=*/halo_s, /*data_dim0=*/halo_e,\
                        /*send_for_dim2_start=*/0, /*send_for_dim2_end=*/dim2,\
                        /*send_for_dim1_start=*/halo_n, /*send_for_dim1_end=*/halo_n+halo_s,\
                        /*send_for_dim_offset=*/halo_w,\
                        /*recv_for_dim2_start=*/0, /*recv_for_dim2_end=*/dim2,\
                        /*recv_for_dim1_start=*/halo_n+core_dim1, /*recv_for_dim1_end=*/dim1,\
                        /*recv_for_dim0_offset=*/halo_w+core_dim0);\
    }\
    /* 6. 向邻居mes发送, 从邻居mwn接收, tag = 18 */\
    if (dim2*halo_n*halo_w != 0) {\
        exchange_data_3D_##TYPE(input, 18, dim1, dim0,\
                        /*a_index2=*/index2, /*a_index1=*/index1+1, /*a_index0=*/index0+1,\
                        /*b_index2=*/index2, /*b_index1=*/index1-1, /*b_index0=*/index0-1,\
                        mpi_dim2, mpi_dim1, mpi_dim0,\
                        /*data_dim2=*/dim2, /*data_dim1=*/halo_n, /*data_dim0=*/halo_w,\
                        /*send_for_dim2_start=*/0, /*send_for_dim2_end=*/dim2,\
                        /*send_for_dim1_start=*/core_dim1, /*send_for_dim1_end=*/core_dim1+halo_n,\
                        /*send_for_dim_offset=*/core_dim0,\
                        /*recv_for_dim2_start=*/0, /*recv_for_dim2_end=*/dim2,\
                        /*recv_for_dim1_start=*/0, /*recv_for_dim1_end=*/halo_n,\
                        /*recv_for_dim0_offset=*/0);\
    }\
    /* 7. 向邻居men发送, 从邻居mws接收, tag = 12 */\
    if (dim2*halo_s*halo_w != 0) {\
        exchange_data_3D_##TYPE(input, 12, dim1, dim0,\
                        /*a_index2=*/index2, /*a_index1=*/index1-1, /*a_index0=*/index0+1,\
                        /*b_index2=*/index2, /*b_index1=*/index1+1, /*b_index0=*/index0-1,\
                        mpi_dim2, mpi_dim1, mpi_dim0,\
                        /*data_dim2=*/dim2, /*data_dim1=*/halo_s, /*data_dim0=*/halo_w,\
                        /*send_for_dim2_start=*/0, /*send_for_dim2_end=*/dim2,\
                        /*send_for_dim1_start=*/halo_n, /*send_for_dim1_end=*/halo_n+halo_s,\
                        /*send_for_dim_offset=*/core_dim0,\
                        /*recv_for_dim2_start=*/0, /*recv_for_dim2_end=*/dim2,\
                        /*recv_for_dim1_start=*/halo_n+core_dim1, /*recv_for_dim1_end=*/dim1,\
                        /*recv_for_dim0_offset=*/0);\
    }\
    /* 8. 向邻居mws发送, 从邻居men接收, tag = 16 */\
    if (dim2*halo_n*halo_e != 0) {\
        exchange_data_3D_##TYPE(input, 16, dim1, dim0,\
                        /*a_index2=*/index2, /*a_index1=*/index1+1, /*a_index0=*/index0-1,\
                        /*b_index2=*/index2, /*b_index1=*/index1-1, /*b_index0=*/index0+1,\
                        mpi_dim2, mpi_dim1, mpi_dim0,\
                        /*data_dim2=*/dim2, /*data_dim1=*/halo_n, /*data_dim0=*/halo_e,\
                        /*send_for_dim2_start=*/0, /*send_for_dim2_end=*/dim2,\
                        /*send_for_dim1_start=*/core_dim1, /*send_for_dim1_end=*/core_dim1+halo_n,\
                        /*send_for_dim_offset=*/halo_w,\
                        /*recv_for_dim2_start=*/0, /*recv_for_dim2_end=*/dim2,\
                        /*recv_for_dim1_start=*/0, /*recv_for_dim1_end=*/halo_n,\
                        /*recv_for_dim0_offset=*/core_dim0+halo_w);\
    }\
    /* 9. 向邻居tn发送, 从邻居bs接收, tag = 21 */\
    if (halo_b*halo_s*dim0 != 0) {\
        exchange_data_3D_##TYPE(input, 21, dim1, dim0,\
                        /*a_index2=*/index2-1, /*a_index1=*/index1-1, /*a_index0=*/index0,\
                        /*b_index2=*/index2+1, /*b_index1=*/index1+1, /*b_index0=*/index0,\
                        mpi_dim2, mpi_dim1, mpi_dim0,\
                        /*data_dim2=*/halo_b, /*data_dim1=*/halo_s, /*data_dim0=*/dim0,\
                        /*send_for_dim2_start=*/halo_t, /*send_for_dim2_end=*/halo_t+halo_b,\
                        /*send_for_dim1_start=*/halo_n, /*send_for_dim1_end=*/halo_n+halo_s,\
                        /*send_for_dim_offset=*/0,\
                        /*recv_for_dim2_start=*/core_dim2+halo_t, /*recv_for_dim2_end=*/dim2,\
                        /*recv_for_dim1_start=*/core_dim1+halo_n, /*recv_for_dim1_end=*/dim1,\
                        /*recv_for_dim0_offset=*/0);\
    }\
    /* 10. 向邻居bs发送, 从邻居tn接收, tag = 7 */\
    if (halo_t*halo_n*dim0 != 0) {\
        exchange_data_3D_##TYPE(input, 7, dim1, dim0,\
                        /*a_index2=*/index2+1, /*a_index1=*/index1+1, /*a_index0=*/index0,\
                        /*b_index2=*/index2-1, /*b_index1=*/index1-1, /*b_index0=*/index0,\
                        mpi_dim2, mpi_dim1, mpi_dim0,\
                        /*data_dim2=*/halo_t, /*data_dim1=*/halo_n, /*data_dim0=*/dim0,\
                        /*send_for_dim2_start=*/core_dim2, /*send_for_dim2_end=*/halo_t+core_dim2,\
                        /*send_for_dim1_start=*/core_dim1, /*send_for_dim1_end=*/halo_n+core_dim1,\
                        /*send_for_dim_offset=*/0,\
                        /*recv_for_dim2_start=*/0, /*recv_for_dim2_end=*/halo_t,\
                        /*recv_for_dim1_start=*/0, /*recv_for_dim1_end=*/halo_n,\
                        /*recv_for_dim0_offset=*/0);\
    }\
    /* 11. 向邻居ts发送, 从邻居bn接收, tag = 27 */\
    if (halo_b*halo_n*dim0 != 0) {\
        exchange_data_3D_##TYPE(input, 27, dim1, dim0,\
                        /*a_index2=*/index2-1, /*a_index1=*/index1+1, /*a_index0=*/index0,\
                        /*b_index2=*/index2+1, /*b_index1=*/index1-1, /*b_index0=*/index0,\
                        mpi_dim2, mpi_dim1, mpi_dim0,\
                        /*data_dim2=*/halo_b, /*data_dim1=*/halo_n, /*data_dim0=*/dim0,\
                        /*send_for_dim2_start=*/halo_t, /*send_for_dim2_end=*/halo_t+halo_b,\
                        /*send_for_dim1_start=*/core_dim1, /*send_for_dim1_end=*/halo_n+core_dim1,\
                        /*send_for_dim_offset=*/0,\
                        /*recv_for_dim2_start=*/halo_t+core_dim2, /*recv_for_dim2_end=*/dim2,\
                        /*recv_for_dim1_start=*/0, /*recv_for_dim1_end=*/halo_n,\
                        /*recv_for_dim0_offset=*/0);\
    }\
    /* 12. 向邻居bn发送, 从邻居ts接收, tag = 1 */\
    if (halo_t*halo_s*dim0 != 0) {\
        exchange_data_3D_##TYPE(input, 1, dim1, dim0,\
                        /*a_index2=*/index2+1, /*a_index1=*/index1+1, /*a_index0=*/index0,\
                        /*b_index2=*/index2-1, /*b_index1=*/index1-1, /*b_index0=*/index0,\
                        mpi_dim2, mpi_dim1, mpi_dim0,\
                        /*data_dim2=*/halo_t, /*data_dim1=*/halo_s, /*data_dim0=*/dim0,\
                        /*send_for_dim2_start=*/core_dim2, /*send_for_dim2_end=*/halo_t+core_dim2,\
                        /*send_for_dim1_start=*/halo_n, /*send_for_dim1_end=*/halo_n+halo_s,\
                        /*send_for_dim_offset=*/0,\
                        /*recv_for_dim2_start=*/0, /*recv_for_dim2_end=*/halo_t,\
                        /*recv_for_dim1_start=*/halo_n+core_dim1, /*recv_for_dim1_end=*/dim1,\
                        /*recv_for_dim0_offset=*/0);\
    }\
    /* 13. 向邻居tw发送, 从邻居be接收, tag = 23 */\
    if (halo_b*dim1*halo_e != 0) {\
        exchange_data_3D_##TYPE(input, 23, dim1, dim0,\
                        /*a_index2=*/index2-1, /*a_index1=*/index1, /*a_index0=*/index0-1,\
                        /*b_index2=*/index2+1, /*b_index1=*/index1, /*b_index0=*/index0+1,\
                        mpi_dim2, mpi_dim1, mpi_dim0,\
                        /*data_dim2=*/halo_b, /*data_dim1=*/dim1, /*data_dim0=*/halo_e,\
                        /*send_for_dim2_start=*/halo_t, /*send_for_dim2_end=*/halo_t+halo_b,\
                        /*send_for_dim1_start=*/0, /*send_for_dim1_end=*/dim1,\
                        /*send_for_dim_offset=*/halo_w,\
                        /*recv_for_dim2_start=*/halo_t+core_dim2, /*recv_for_dim2_end=*/dim2,\
                        /*recv_for_dim1_start=*/0, /*recv_for_dim1_end=*/dim1,\
                        /*recv_for_dim0_offset=*/halo_w+core_dim0);\
    }\
    /* 14. 向邻居be发送, 从邻居tw接收, tag = 5 */\
    if (halo_t*dim1*halo_w != 0) {\
        exchange_data_3D_##TYPE(input, 5, dim1, dim0,\
                        /*a_index2=*/index2+1, /*a_index1=*/index1, /*a_index0=*/index0+1,\
                        /*b_index2=*/index2-1, /*b_index1=*/index1, /*b_index0=*/index0-1,\
                        mpi_dim2, mpi_dim1, mpi_dim0,\
                        /*data_dim2=*/halo_t, /*data_dim1=*/dim1, /*data_dim0=*/halo_w,\
                        /*send_for_dim2_start=*/core_dim2, /*send_for_dim2_end=*/halo_t+core_dim2,\
                        /*send_for_dim1_start=*/0, /*send_for_dim1_end=*/dim1,\
                        /*send_for_dim_offset=*/core_dim0,\
                        /*recv_for_dim2_start=*/0, /*recv_for_dim2_end=*/halo_t,\
                        /*recv_for_dim1_start=*/0, /*recv_for_dim1_end=*/dim1,\
                        /*recv_for_dim0_offset=*/0);\
    }\
    /* 15. 向邻居te发送, 从邻居bw接收, tag = 25 */\
    if (halo_b*dim1*halo_w != 0) {\
        exchange_data_3D_##TYPE(input, 25, dim1, dim0,\
                        /*a_index2=*/index2-1, /*a_index1=*/index1, /*a_index0=*/index0+1,\
                        /*b_index2=*/index2+1, /*b_index1=*/index1, /*b_index0=*/index0-1,\
                        mpi_dim2, mpi_dim1, mpi_dim0,\
                        /*data_dim2=*/halo_b, /*data_dim1=*/dim1, /*data_dim0=*/halo_w,\
                        /*send_for_dim2_start=*/halo_t, /*send_for_dim2_end=*/halo_t+halo_b,\
                        /*send_for_dim1_start=*/0, /*send_for_dim1_end=*/dim1,\
                        /*send_for_dim_offset=*/core_dim0,\
                        /*recv_for_dim2_start=*/halo_t+core_dim2, /*recv_for_dim2_end=*/dim2,\
                        /*recv_for_dim1_start=*/0, /*recv_for_dim1_end=*/dim1,\
                        /*recv_for_dim0_offset=*/0);\
    }\
    /* 16. 向邻居bw发送, 从邻居te接收, tag = 3 */\
    if (halo_t*dim1*halo_e != 0) {\
        exchange_data_3D_##TYPE(input, 3, dim1, dim0,\
                        /*a_index2=*/index2+1, /*a_index1=*/index1, /*a_index0=*/index0-1,\
                        /*b_index2=*/index2-1, /*b_index1=*/index1, /*b_index0=*/index0+1,\
                        mpi_dim2, mpi_dim1, mpi_dim0,\
                        /*data_dim2=*/halo_b, /*data_dim1=*/dim1, /*data_dim0=*/halo_w,\
                        /*send_for_dim2_start=*/core_dim2, /*send_for_dim2_end=*/halo_t+core_dim2,\
                        /*send_for_dim1_start=*/0, /*send_for_dim1_end=*/dim1,\
                        /*send_for_dim_offset=*/halo_w,\
                        /*recv_for_dim2_start=*/0, /*recv_for_dim2_end=*/halo_t,\
                        /*recv_for_dim1_start=*/0, /*recv_for_dim1_end=*/dim1,\
                        /*recv_for_dim0_offset=*/halo_w+core_dim0);\
    }\
    /* 17. 向邻居tc发送, 从邻居bc接收, tag = 24 */\
    if (halo_b*dim1*dim0 != 0) {\
        exchange_data_3D_##TYPE(input, 24, dim1, dim0,\
                        /*a_index2=*/index2-1, /*a_index1=*/index1, /*a_index0=*/index0,\
                        /*b_index2=*/index2+1, /*b_index1=*/index1, /*b_index0=*/index0,\
                        mpi_dim2, mpi_dim1, mpi_dim0,\
                        /*data_dim2=*/halo_b, /*data_dim1=*/dim1, /*data_dim0=*/dim0,\
                        /*send_for_dim2_start=*/halo_t, /*send_for_dim2_end=*/halo_t+halo_b,\
                        /*send_for_dim1_start=*/0, /*send_for_dim1_end=*/dim1,\
                        /*send_for_dim_offset=*/0,\
                        /*recv_for_dim2_start=*/halo_t+core_dim2, /*recv_for_dim2_end=*/dim2,\
                        /*recv_for_dim1_start=*/0, /*recv_for_dim1_end=*/dim1,\
                        /*recv_for_dim0_offset=*/0);\
    }\
    /* 18. 向邻居bc发送, 从邻居tc接收, tag = 4 */\
    if (halo_t*dim1*dim0 != 0) {\
        exchange_data_3D_##TYPE(input, 4, dim1, dim0,\
                        /*a_index2=*/index2+1, /*a_index1=*/index1, /*a_index0=*/index0,\
                        /*b_index2=*/index2-1, /*b_index1=*/index1, /*b_index0=*/index0,\
                        mpi_dim2, mpi_dim1, mpi_dim0,\
                        /*data_dim2=*/halo_t, /*data_dim1=*/dim1, /*data_dim0=*/dim0,\
                        /*send_for_dim2_start=*/core_dim2, /*send_for_dim2_end=*/halo_t+core_dim2,\
                        /*send_for_dim1_start=*/0, /*send_for_dim1_end=*/dim1,\
                        /*send_for_dim_offset=*/0,\
                        /*recv_for_dim2_start=*/0, /*recv_for_dim2_end=*/halo_t,\
                        /*recv_for_dim1_start=*/0, /*recv_for_dim1_end=*/dim1,\
                        /*recv_for_dim0_offset=*/0);\
    }\
    /* 19. 向邻居twn发送, 从邻居bes接收, tag = 20 */\
    if (halo_b*halo_s*halo_e != 0) {\
        exchange_data_3D_##TYPE(input, 20, dim1, dim0,\
                        /*a_index2=*/index2-1, /*a_index1=*/index1-1, /*a_index0=*/index0-1,\
                        /*b_index2=*/index2+1, /*b_index1=*/index1+1, /*b_index0=*/index0+1,\
                        mpi_dim2, mpi_dim1, mpi_dim0,\
                        /*data_dim2=*/halo_b, /*data_dim1=*/halo_s, /*data_dim0=*/halo_e,\
                        /*send_for_dim2_start=*/halo_t, /*send_for_dim2_end=*/halo_t+halo_b,\
                        /*send_for_dim1_start=*/halo_n, /*send_for_dim1_end=*/halo_w+halo_s,\
                        /*send_for_dim_offset=*/halo_w,\
                        /*recv_for_dim2_start=*/halo_t+core_dim2, /*recv_for_dim2_end=*/dim2,\
                        /*recv_for_dim1_start=*/halo_n+core_dim1, /*recv_for_dim1_end=*/dim1,\
                        /*recv_for_dim0_offset=*/halo_w+core_dim0);\
    }\
    /* 20. 向邻居bes发送, 从邻居twn接收, tag = 28 */\
    if (halo_t*halo_n*halo_w != 0) {\
        exchange_data_3D_##TYPE(input, 28, dim1, dim0,\
                        /*a_index2=*/index2+1, /*a_index1=*/index1+1, /*a_index0=*/index0+1,\
                        /*b_index2=*/index2-1, /*b_index1=*/index1-1, /*b_index0=*/index0-1,\
                        mpi_dim2, mpi_dim1, mpi_dim0,\
                        /*data_dim2=*/halo_t, /*data_dim1=*/halo_n, /*data_dim0=*/halo_w,\
                        /*send_for_dim2_start=*/core_dim2, /*send_for_dim2_end=*/halo_t+core_dim2,\
                        /*send_for_dim1_start=*/core_dim1, /*send_for_dim1_end=*/halo_n+core_dim1,\
                        /*send_for_dim_offset=*/core_dim0,\
                        /*recv_for_dim2_start=*/0, /*recv_for_dim2_end=*/halo_t,\
                        /*recv_for_dim1_start=*/0, /*recv_for_dim1_end=*/halo_n,\
                        /*recv_for_dim0_offset=*/0);\
    }\
    /* 21. 向邻居ten发送, 从邻居bws接收, tag = 22 */\
    if (halo_b*halo_s*halo_w != 0) {\
        exchange_data_3D_##TYPE(input, 22, dim1, dim0,\
                        /*a_index2=*/index2-1, /*a_index1=*/index1-1, /*a_index0=*/index0+1,\
                        /*b_index2=*/index2+1, /*b_index1=*/index1+1, /*b_index0=*/index0-1,\
                        mpi_dim2, mpi_dim1, mpi_dim0,\
                        /*data_dim2=*/halo_b, /*data_dim1=*/halo_s, /*data_dim0=*/halo_w,\
                        /*send_for_dim2_start=*/halo_t, /*send_for_dim2_end=*/halo_t+halo_b,\
                        /*send_for_dim1_start=*/halo_n, /*send_for_dim1_end=*/halo_n+halo_s,\
                        /*send_for_dim_offset=*/core_dim0,\
                        /*recv_for_dim2_start=*/halo_t+core_dim2, /*recv_for_dim2_end=*/dim2,\
                        /*recv_for_dim1_start=*/halo_n+core_dim1, /*recv_for_dim1_end=*/dim1,\
                        /*recv_for_dim0_offset=*/0);\
    }\
    /* 22. 向邻居bws发送, 从邻居ten接收, tag = 6 */\
    if (halo_t*halo_n*halo_e != 0) {\
        exchange_data_3D_##TYPE(input, 6, dim1, dim0,\
                        /*a_index2=*/index2+1, /*a_index1=*/index1+1, /*a_index0=*/index0-1,\
                        /*b_index2=*/index2-1, /*b_index1=*/index1-1, /*b_index0=*/index0+1,\
                        mpi_dim2, mpi_dim1, mpi_dim0,\
                        /*data_dim2=*/halo_t, /*data_dim1=*/halo_n, /*data_dim0=*/halo_e,\
                        /*send_for_dim2_start=*/core_dim2, /*send_for_dim2_end=*/halo_t+core_dim2,\
                        /*send_for_dim1_start=*/core_dim1, /*send_for_dim1_end=*/halo_n+core_dim1,\
                        /*send_for_dim_offset=*/halo_w,\
                        /*recv_for_dim2_start=*/0, /*recv_for_dim2_end=*/halo_t,\
                        /*recv_for_dim1_start=*/0, /*recv_for_dim1_end=*/halo_n,\
                        /*recv_for_dim0_offset=*/halo_w+core_dim0);\
    }\
    /* 23. 向邻居tws发送, 从邻居ben接收, tag = 26 */\
    if (halo_b*halo_n*halo_e != 0) {\
        exchange_data_3D_##TYPE(input, 26, dim1, dim0,\
                        /*a_index2=*/index2-1, /*a_index1=*/index1+1, /*a_index0=*/index0-1,\
                        /*b_index2=*/index2+1, /*b_index1=*/index1-1, /*b_index0=*/index0+1,\
                        mpi_dim2, mpi_dim1, mpi_dim0,\
                        /*data_dim2=*/halo_b, /*data_dim1=*/halo_n, /*data_dim0=*/halo_e,\
                        /*send_for_dim2_start=*/halo_t, /*send_for_dim2_end=*/halo_t+halo_b,\
                        /*send_for_dim1_start=*/core_dim1, /*send_for_dim1_end=*/halo_n+core_dim1,\
                        /*send_for_dim_offset=*/halo_w,\
                        /*recv_for_dim2_start=*/halo_t+core_dim2, /*recv_for_dim2_end=*/dim2,\
                        /*recv_for_dim1_start=*/0, /*recv_for_dim1_end=*/halo_n,\
                        /*recv_for_dim0_offset=*/halo_w+core_dim0);\
    }\
    /* 24. 向邻居ben发送, 从邻居tws接收, tag = 2 */\
    if (halo_t*halo_s*halo_w != 0) {\
        exchange_data_3D_##TYPE(input, 2, dim1, dim0,\
                        /*a_index2=*/index2+1, /*a_index1=*/index1-1, /*a_index0=*/index0+1,\
                        /*b_index2=*/index2-1, /*b_index1=*/index1+1, /*b_index0=*/index0-1,\
                        mpi_dim2, mpi_dim1, mpi_dim0,\
                        /*data_dim2=*/halo_t, /*data_dim1=*/halo_s, /*data_dim0=*/halo_w,\
                        /*send_for_dim2_start=*/core_dim2, /*send_for_dim2_end=*/halo_t+core_dim2,\
                        /*send_for_dim1_start=*/halo_n, /*send_for_dim1_end=*/halo_n+halo_s,\
                        /*send_for_dim_offset=*/core_dim0,\
                        /*recv_for_dim2_start=*/0, /*recv_for_dim2_end=*/halo_t,\
                        /*recv_for_dim1_start=*/halo_n+core_dim1, /*recv_for_dim1_end=*/dim1,\
                        /*recv_for_dim0_offset=*/0);\
    }\
    /* 25. 向邻居tes发送, 从邻居bwn接收, tag = 28 */\
    if (halo_b*halo_n*halo_w != 0) {\
        exchange_data_3D_##TYPE(input, 28, dim1, dim0,\
                        /*a_index2=*/index2-1, /*a_index1=*/index1+1, /*a_index0=*/index0+1,\
                        /*b_index2=*/index2+1, /*b_index1=*/index1-1, /*b_index0=*/index0-1,\
                        mpi_dim2, mpi_dim1, mpi_dim0,\
                        /*data_dim2=*/halo_b, /*data_dim1=*/halo_n, /*data_dim0=*/halo_w,\
                        /*send_for_dim2_start=*/halo_t, /*send_for_dim2_end=*/halo_t+halo_b,\
                        /*send_for_dim1_start=*/core_dim1, /*send_for_dim1_end=*/halo_n+core_dim1,\
                        /*send_for_dim_offset=*/core_dim0,\
                        /*recv_for_dim2_start=*/halo_t+core_dim2, /*recv_for_dim2_end=*/dim2,\
                        /*recv_for_dim1_start=*/0, /*recv_for_dim1_end=*/halo_n,\
                        /*recv_for_dim0_offset=*/0);\
    }\
    /* 26. 向邻居bwn发送, 从邻居tes接收, tag = 0 */\
    if (halo_t*halo_s*halo_e != 0) {\
        exchange_data_3D_##TYPE(input, 0, dim1, dim0,\
                        /*a_index2=*/index2+1, /*a_index1=*/index1-1, /*a_index0=*/index0-1,\
                        /*b_index2=*/index2-1, /*b_index1=*/index1+1, /*b_index0=*/index0+1,\
                        mpi_dim2, mpi_dim1, mpi_dim0,\
                        /*data_dim2=*/halo_t, /*data_dim1=*/halo_s, /*data_dim0=*/halo_e,\
                        /*send_for_dim2_start=*/core_dim2, /*send_for_dim2_end=*/halo_t+core_dim2,\
                        /*send_for_dim1_start=*/halo_n, /*send_for_dim1_end=*/halo_n+halo_s,\
                        /*send_for_dim_offset=*/halo_w,\
                        /*recv_for_dim2_start=*/0, /*recv_for_dim2_end=*/halo_t,\
                        /*recv_for_dim1_start=*/halo_n+core_dim1, /*recv_for_dim1_end=*/dim1,\
                        /*recv_for_dim0_offset=*/halo_w+core_dim0);\
    }\
}

// 获取本从核在通信域中的rank
int mpiGetMyRank() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

EXCHANGE_DATA_2D(double, MPI_DOUBLE)
EXCHANGE_DATA_2D(float, MPI_FLOAT)
EXCHANGE_HALO_2D(double)
EXCHANGE_HALO_2D(float)

EXCHANGE_DATA_3D(double, MPI_DOUBLE)
EXCHANGE_DATA_3D(float, MPI_FLOAT)
EXCHANGE_HALO_3D(double)
EXCHANGE_HALO_3D(float)
