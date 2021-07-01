/**
 * @file mpi_io.h
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief 本文件定义了数据加载和数据收集等函数, 提供给用户使用
 * 该部分函数仅在通信域中的0号进程中使用
 * @version 0.1
 * @date 2021-06-26
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#ifndef _SWSTEN_MPI_IO_H_
#define _SWSTEN_MPI_IO_H_

#include <mpi.h>

/****************************** 二维情况 ***************************************/
/* 0 号进程根据其他进程的位置发送相应的数据 */
#define SEND_DATA_2D_DECLARE(TYPE, MPI_TYPE)\
void swsten_send_data_2D_##TYPE(TYPE *sendArray, int sendArrayDim1, int sendArrayDim0,\
                         TYPE *recvArray, int recvArrayDim1, int recvArrayDim0,\
                         int halo_n, int halo_s, int halo_w, int halo_e,\
                         int mpiDim1, int mpiDim0);

/* 其他进程向0号进程发送相应的数据 */
#define RECV_DATA_2D_DECLARE(TYPE, MPI_TYPE)\
void swsten_recv_data_2D_##TYPE(TYPE *recvArray, int recvArrayDim1, int recvArrayDim0,\
                         TYPE *sendArray, int sendArrayDim1, int sendArrayDim0,\
                         int halo_n, int halo_s, int halo_w, int halo_e,\
                         int mpiDim1, int mpiDim0);

/****************************** 三维情况 ***************************************/
/* 0 号进程根据其他进程的位置发送相应的数据 */
#define SEND_DATA_3D_DECLARE(TYPE, MPI_TYPE)\
void swsten_send_data_3D_##TYPE(TYPE *sendArray,\
                         int sendArrayDim2, int sendArrayDim1, int sendArrayDim0,\
                         TYPE *recvArray,\
                         int recvArrayDim2, int recvArrayDim1, int recvArrayDim0,\
                         int halo_t, int halo_b, int halo_n, int halo_s, int halo_w, int halo_e,\
                         int mpiDim2, int mpiDim1, int mpiDim0);

/* 其他进程向0号进程发送相应的数据 */
#define RECV_DATA_3D_DECLARE(TYPE, MPI_TYPE)\
void swsten_recv_data_3D_##TYPE(TYPE *recvArray, int recvArrayDim2, int recvArrayDim1, int recvArrayDim0,\
                         TYPE *sendArray, int sendArrayDim2, int sendArrayDim1, int sendArrayDim0,\
                         int halo_t, int halo_b, int halo_n, int halo_s, int halo_w, int halo_e,\
                         int mpiDim2, int mpiDim1, int mpiDim0);


SEND_DATA_2D_DECLARE(float, MPI_FLOAT);
SEND_DATA_2D_DECLARE(double, MPI_DOUBLE);
RECV_DATA_2D_DECLARE(float, MPI_FLOAT);
RECV_DATA_2D_DECLARE(double, MPI_DOUBLE);

SEND_DATA_3D_DECLARE(float, MPI_FLOAT);
SEND_DATA_3D_DECLARE(double, MPI_DOUBLE);
RECV_DATA_3D_DECLARE(float, MPI_FLOAT);
RECV_DATA_3D_DECLARE(double, MPI_DOUBLE);

#endif /* end of _SWSTEN_MPI_IO_H_ */
