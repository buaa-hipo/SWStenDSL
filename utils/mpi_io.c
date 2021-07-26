/**
 * @file mpi_io.c
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2021-07-10
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi_io.h"

/****************************** 二维情况 ***************************************/
/* 将指定合适位置的数据从sendArray加载到recv_buffer中 */
#define LOAD_DATA_2D(TYPE)\
void swsten_load_data_2D_##TYPE(TYPE *sendArray, int sendArrayDim0,\
                         TYPE *recv_buffer, int width,\
                         int start, int end, int offset)\
{\
    int cnt = 0;\
    int i;\
    for (i = start; i < end; i++) {\
        memcpy(&recv_buffer[cnt], &sendArray[i*sendArrayDim0 + offset], width*sizeof(TYPE));\
        cnt += width;\
    }\
}

/* 将send_buffer中的数据加载到recvArray的对应位置上 */
#define STORE_DATA_2D(TYPE)\
void swsten_store_data_2D_##TYPE(TYPE *send_buffer, int sendArrayDim0,\
                          TYPE *recvArray, int recvArrayDim0,\
                          int data_height, int data_width,\
                          int halo_n, int halo_w, int mpi_i, int mpi_j)\
{\
    int i;\
    for (i = 0; i < data_height; i++) {\
        memcpy(&recvArray[(mpi_i*data_height+halo_n + i)*recvArrayDim0 + (mpi_j*data_width+halo_w)],\
               &send_buffer[(halo_n+i)*sendArrayDim0 + halo_w], data_width*sizeof(TYPE));\
    }\
}

/* 0 号进程根据其他进程的位置发送相应的数据 */
#define SEND_DATA_2D(TYPE, MPI_TYPE)\
void swsten_send_data_2D_##TYPE(TYPE *sendArray, int sendArrayDim1, int sendArrayDim0,\
                         TYPE *recvArray, int recvArrayDim1, int recvArrayDim0,\
                         int halo_n, int halo_s, int halo_w, int halo_e,\
                         int mpiDim1, int mpiDim0)\
{\
    /* get mpi rank */\
    int my_rank = 0;\
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);\
    int data_width = recvArrayDim0 - halo_w - halo_e;\
    int data_height = recvArrayDim1 - halo_n - halo_s;\
    /* 0 号进程分发数据 */\
    if (my_rank == 0) {\
        /* 发送buffer */\
        TYPE *send_buffer = (TYPE *)malloc(recvArrayDim1*recvArrayDim0*sizeof(TYPE));\
        int mpi_i, mpi_j;\
        for (mpi_i = 0; mpi_i < mpiDim1; mpi_i++) {\
            for (mpi_j = 0; mpi_j < mpiDim0; mpi_j++) {\
                /* 计算进程号 */\
                int pid = mpi_i*mpiDim0 + mpi_j;\
                if (pid == 0) {\
                    swsten_load_data_2D_##TYPE(sendArray, sendArrayDim0,\
                                        recvArray, recvArrayDim0,\
                                        0, recvArrayDim1, 0);\
                } else {\
                    swsten_load_data_2D_##TYPE(sendArray, sendArrayDim0,\
                                        send_buffer, recvArrayDim0,\
                                        mpi_i*data_height, (mpi_i+1)*data_height+halo_n+halo_s, mpi_j*data_width);\
                    /* 将数据发送出去 */\
                    MPI_Send(send_buffer, recvArrayDim1*recvArrayDim0, MPI_TYPE, pid, 0, MPI_COMM_WORLD);\
                }\
            }\
        }\
        free(send_buffer);\
        send_buffer = NULL;\
    } else {\
        /* 非0号进程接收数据 */\
        MPI_Recv(recvArray, recvArrayDim1*recvArrayDim0, MPI_TYPE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);\
    }\
}

/* 其他进程向0号进程发送相应的数据 */
#define RECV_DATA_2D(TYPE, MPI_TYPE)\
void swsten_recv_data_2D_##TYPE(TYPE *recvArray, int recvArrayDim1, int recvArrayDim0,\
                         TYPE *sendArray, int sendArrayDim1, int sendArrayDim0,\
                         int halo_n, int halo_s, int halo_w, int halo_e,\
                         int mpiDim1, int mpiDim0)\
{\
    /* get mpi rank and size */\
    int my_rank = 0;\
    int comm_size = 0;\
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);\
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);\
    int data_width = sendArrayDim0 - halo_w - halo_e;\
    int data_height = sendArrayDim1 - halo_n - halo_s;\
    MPI_Status status;\
    /* 0 号进程接收数据 */\
    if (my_rank == 0) {\
        /* 接收buffer */\
        TYPE *recv_buffer = (TYPE *)malloc(sendArrayDim1*sendArrayDim0*sizeof(TYPE));\
        /* 0 号进程先将自己的数据写回到相应的位置上 */\
        swsten_store_data_2D_##TYPE(sendArray, sendArrayDim0, recvArray, recvArrayDim0,\
                             data_height, data_width, halo_n, halo_w, 0, 0);\
        int i;\
        for (i = 1; i < comm_size; i++) {\
            MPI_Recv(recv_buffer, sendArrayDim1*sendArrayDim0, MPI_TYPE,\
                     MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);\
            /* 获取接受到数据的发送端 */\
            int mpi_pid =  status.MPI_SOURCE;\
            int mpi_i = mpi_pid / mpiDim0;\
            int mpi_j = mpi_pid % mpiDim0;\
            swsten_store_data_2D_##TYPE(recv_buffer, sendArrayDim0, recvArray, recvArrayDim0,\
                                 data_height, data_width, halo_n, halo_w,\
                                 mpi_i, mpi_j);\
        }\
        free(recv_buffer);\
        recv_buffer = NULL;\
    } else {\
        /* 非0号进程向0号进程发送数据 */\
        MPI_Send(sendArray, sendArrayDim1*sendArrayDim0, MPI_TYPE, 0, 0, MPI_COMM_WORLD);\
    }\
}

/******************************* 三维情况 **************************************/
/* 将指定合适位置的数据从sendArray加载到recv_buffer中 */
#define LOAD_DATA_3D(TYPE)\
void swsten_load_data_3D_##TYPE(TYPE *sendArray, int sendArrayDim1, int sendArrayDim0,\
                         TYPE *recv_buffer, int width,\
                         int dim2_start, int dim2_end,\
                         int dim1_start, int dim1_end, int dim0_offset)\
{\
    int cnt = 0;\
    int i, j;\
    for (i = dim2_start; i < dim2_end; i++) {\
        for (j = dim1_start; j < dim1_end; j++) {\
            memcpy(&recv_buffer[cnt],\
                   &sendArray[i*sendArrayDim1*sendArrayDim0 + j*sendArrayDim0 + dim0_offset],\
                   width*sizeof(TYPE));\
            cnt += width;\
        }\
    }\
}

/* 将send_buffer中的数据加载到recvArray的对应位置上 */
#define STORE_DATA_3D(TYPE)\
void swsten_store_data_3D_##TYPE(TYPE *send_buffer, int sendArrayDim1, int sendArrayDim0,\
                          TYPE *recvArray, int recvArrayDim1, int recvArrayDim0,\
                          int data_dim2, int data_dim1, int data_dim0,\
                          int halo_t, int halo_n, int halo_w,\
                          int mpi_i, int mpi_j, int mpi_k)\
{\
    int i, j;\
    for (i = 0; i < data_dim2; i++) {\
        for (j = 0; j < data_dim1; j++) {\
            memcpy(&recvArray[(mpi_i*data_dim2+halo_t+i)*recvArrayDim1*recvArrayDim0 + (mpi_j*data_dim1+halo_n+j)*recvArrayDim0 + (mpi_k*data_dim0+halo_w)],\
                   &send_buffer[(halo_t+i)*sendArrayDim1*sendArrayDim0 + (halo_n+j)*sendArrayDim0 + halo_w],\
                   data_dim0*sizeof(TYPE));\
        }\
    }\
}

/* 0 号进程根据其他进程的位置发送相应的数据 */
#define SEND_DATA_3D(TYPE, MPI_TYPE)\
void swsten_send_data_3D_##TYPE(TYPE *sendArray,\
                         int sendArrayDim2, int sendArrayDim1, int sendArrayDim0,\
                         TYPE *recvArray,\
                         int recvArrayDim2, int recvArrayDim1, int recvArrayDim0,\
                         int halo_t, int halo_b, int halo_n, int halo_s, int halo_w, int halo_e,\
                         int mpiDim2, int mpiDim1, int mpiDim0)\
{\
    /* get mpi rank */\
    int my_rank = 0;\
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);\
    int data_dim2 = recvArrayDim2 - halo_t - halo_b;\
    int data_dim1 = recvArrayDim1 - halo_n - halo_s;\
    int data_dim0 = recvArrayDim0 - halo_w - halo_e;\
    /* 0 号进程分发数据 */\
    if (my_rank == 0) {\
        /* 发送buffer */\
        TYPE *send_buffer = (TYPE *)malloc(recvArrayDim2*recvArrayDim1*recvArrayDim0*sizeof(TYPE));\
        int mpi_i, mpi_j, mpi_k;\
        for (mpi_i = 0; mpi_i < mpiDim2; mpi_i++) {\
            for (mpi_j = 0; mpi_j < mpiDim1; mpi_j++) {\
                for (mpi_k = 0; mpi_k < mpiDim0; mpi_k++) {\
                    /* 计算进程号 */\
                    int pid = mpi_i*mpiDim1*mpiDim0 + mpi_j*mpiDim0 + mpi_k;\
                    if (pid == 0) {\
                        swsten_load_data_3D_##TYPE(sendArray, sendArrayDim1, sendArrayDim0,\
                                            recvArray, recvArrayDim0,\
                                            0, recvArrayDim2,\
                                            0, recvArrayDim1, 0);\
                    } else {\
                        swsten_load_data_3D_##TYPE(sendArray, sendArrayDim1, sendArrayDim0,\
                                            send_buffer, recvArrayDim0,\
                                            mpi_i*data_dim2, (mpi_i+1)*data_dim2+halo_t+halo_b,\
                                            mpi_j*data_dim1, (mpi_j+1)*data_dim1+halo_n+halo_s,\
                                            mpi_k*data_dim0);\
                        /* 将数据发送出去 */\
                        MPI_Send(send_buffer, recvArrayDim2*recvArrayDim1*recvArrayDim0, MPI_TYPE, pid, 0, MPI_COMM_WORLD);\
                    }\
                }\
            }\
        }\
        free(send_buffer);\
        send_buffer = NULL;\
    } else {\
        /* 非0号进程接收数据 */\
        MPI_Recv(recvArray, recvArrayDim2*recvArrayDim1*recvArrayDim0, MPI_TYPE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);\
    }\
}

/* 其他进程向0号进程发送相应的数据 */
#define RECV_DATA_3D(TYPE, MPI_TYPE)\
void swsten_recv_data_3D_##TYPE(TYPE *recvArray, int recvArrayDim2, int recvArrayDim1, int recvArrayDim0,\
                         TYPE *sendArray, int sendArrayDim2, int sendArrayDim1, int sendArrayDim0,\
                         int halo_t, int halo_b, int halo_n, int halo_s, int halo_w, int halo_e,\
                         int mpiDim2, int mpiDim1, int mpiDim0)\
{\
    /* get mpi rank and size */\
    int my_rank = 0;\
    int comm_size = 0;\
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);\
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);\
    int data_dim0 = sendArrayDim0 - halo_w - halo_e;\
    int data_dim1 = sendArrayDim1 - halo_n - halo_s;\
    int data_dim2 = sendArrayDim2 - halo_t - halo_b;\
    MPI_Status status;\
    /* 0号进程接收数据 */\
    if (my_rank == 0) {\
        /* 接收buffer */\
        TYPE *recv_buffer = (TYPE *)malloc(sendArrayDim2*sendArrayDim1*sendArrayDim0*sizeof(TYPE));\
        /* 0号进程先将自己的数据写回到相应位置上 */\
        swsten_store_data_3D_##TYPE(sendArray, sendArrayDim1, sendArrayDim0,\
                             recvArray, recvArrayDim1, recvArrayDim0,\
                             data_dim2, data_dim1, data_dim0,\
                             halo_t, halo_n, halo_w, 0, 0, 0);\
        int i;\
        for (i = 1; i < comm_size; i++) {\
            MPI_Recv(recv_buffer, sendArrayDim2*sendArrayDim1*sendArrayDim0, MPI_TYPE,\
                     MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);\
            /* 获取接收到的数据的发送端 */\
            int mpi_pid = status.MPI_SOURCE;\
            int mpi_i = mpi_pid / (mpiDim1*mpiDim0);\
            int mpi_j = (mpi_pid % (mpiDim1*mpiDim0)) / mpiDim0;\
            int mpi_k = mpi_pid % mpiDim0;\
            swsten_store_data_3D_##TYPE(recv_buffer, sendArrayDim1, sendArrayDim0,\
                                 recvArray, recvArrayDim1, recvArrayDim0,\
                                 data_dim2, data_dim1, data_dim0,\
                                 halo_t, halo_n, halo_w,\
                                 mpi_i, mpi_j, mpi_k);\
        }\
        free(recv_buffer);\
        recv_buffer = NULL;\
    } else {\
        /* 非0号进程向0号进程发送数据 */\
        MPI_Send(sendArray, sendArrayDim2*sendArrayDim1*sendArrayDim0, MPI_TYPE, 0, 0, MPI_COMM_WORLD);\
    }\
}

/* 获取十进制数的位数 */
int get_digit_of_int(int x) 
{
    int res = 0;
    while (x) {
        res += 1;
        x /= 10;
    }
    return res;
}

/* 从文件加载数据*/
#define LOAD_DATA_FROM_FILE(TYPE, FMT)\
void swsten_load_data_from_file_##TYPE(const char *filename, TYPE *array, const int size)\
{\
    int my_rank = 0;\
    int comm_size = 0;\
    int cnt = 0;\
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);\
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);\
    int real_filename_len = strlen(filename) + get_digit_of_int(comm_size);\
    char *real_filename = (char *)malloc(sizeof(char)*real_filename_len);\
    strcpy(real_filename, filename);\
    sprintf(&real_filename[strlen(real_filename)], "%d", my_rank);\
\
    FILE *fp = fopen(real_filename, "r");\
    if (fp == NULL) {\
        printf("WARNING: Failed to open file %s in process %d\n", real_filename, my_rank);\
        printf("Generate random data for compute instead.\n");\
        for (cnt = 0; cnt < size; cnt ++) {\
            TYPE value = rand() % 50;\
            array[cnt] = value;\
        }\
        return;\
    }\
\
    /* read data from file */\
    for (cnt = 0; cnt < size; cnt++) {\
        if (fscanf(fp, FMT, &array[cnt]) <= 0)\
            break;\
    }\
    if (cnt < size)\
        printf("WARNING: From file %s : got %d raw data, which is less than size %d !\n", filename, cnt, size);\
\
    fclose(fp);\
    free(real_filename);\
}

/*将结果写回文件*/
#define STORE_DATA_TO_FILE(TYPE, FMT)\
void swsten_store_data_to_file_##TYPE(const char *filename, TYPE *array, const int size, const int line_len)\
{\
    int my_rank = 0;\
    int comm_size = 0;\
    int cnt = 0;\
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);\
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);\
    int real_filename_len = strlen(filename) + get_digit_of_int(comm_size);\
    char *real_filename = (char *)malloc(sizeof(char)*real_filename_len);\
    strcpy(real_filename, filename);\
    sprintf(&real_filename[strlen(real_filename)], "%d", my_rank);\
\
    FILE *fp = fopen(real_filename, "w");\
    if (fp == NULL) {\
        printf("ERROR: Failed to open file %s in process %d\n", real_filename, my_rank);\
        exit(EXIT_FAILURE);\
    }\
\
    /* store data to file */\
    for (cnt = 0; cnt < size; cnt++) {\
        fprintf(fp, FMT, array[cnt]);\
        if ((cnt+1) % line_len == 0)\
            fprintf(fp, "\n");\
    }\
\
    fclose(fp);\
    free(real_filename);\
}


LOAD_DATA_2D(float)
LOAD_DATA_2D(double)
STORE_DATA_2D(float)
STORE_DATA_2D(double)
SEND_DATA_2D(float, MPI_FLOAT)
SEND_DATA_2D(double, MPI_DOUBLE)
RECV_DATA_2D(float, MPI_FLOAT)
RECV_DATA_2D(double, MPI_DOUBLE)

LOAD_DATA_3D(float)
LOAD_DATA_3D(double)
STORE_DATA_3D(float)
STORE_DATA_3D(double)
SEND_DATA_3D(float, MPI_FLOAT)
SEND_DATA_3D(double, MPI_DOUBLE)
RECV_DATA_3D(float, MPI_FLOAT)
RECV_DATA_3D(double, MPI_DOUBLE)

LOAD_DATA_FROM_FILE(float, "%f")
LOAD_DATA_FROM_FILE(double, "%lf")
STORE_DATA_TO_FILE(float, "%f ")
STORE_DATA_TO_FILE(double, "%lf ")