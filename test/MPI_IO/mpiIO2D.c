#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "../../utils/mpi_io.h"

// 每个节点有一个4x4的数组, 共有2x2个进程, 各个方向的halo为1
double test_array[4][4];
double test_bigArray[6][6];

void print_result(const double *array, int dim1, int dim0) {
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim0; j++)
            printf("%lf ", array[i*dim0+j]);
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int my_rank;
    int comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    if (my_rank == 0) {
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                double value = rand() % 50;
                test_bigArray[i][j] = value;
            }
        }

        printf("The big array:\n");
        print_result((double *)test_bigArray, 6, 6);
    }
    swsten_send_data_2D_double((double *)test_bigArray, 6, 6, (double *)test_array, 4, 4, 1, 1, 1, 1, 2, 2);

    /* 输出发送的结果, 其他进程将结果发回给0号进程, 之后由0号进程输出 */
    if (my_rank != 0) {
        MPI_Send(test_array, 4*4, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    } else {
        printf("Step 1:----------------------------------------------------\n");
        printf("Process %d:++++++++++++++++++++++++++++++++++++++++++++++++\n", my_rank);
        print_result((double *)test_array, 4, 4);

        for (int i = 1; i < comm_size; i++) {
            MPI_Recv(test_array, 4*4, MPI_DOUBLE, i , 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Process %d:+++++++++++++++++++++++++++++++++++++++++++++++++++\n", i);
            print_result((double *)test_array, 4, 4);
        }
    }

    /* 各个进程对test_array进行赋值, 发送给0号进程 */
    for (int i= 1; i < 3; i++) {
        for (int j = 1; j < 3; j++) 
            test_array[i][j] = my_rank;
    }
    swsten_recv_data_2D_double((double *)test_bigArray, 6, 6, (double *)test_array, 4, 4, 1, 1, 1, 1, 2, 2);
    /* 0号进程输出bigArray */
    if (my_rank == 0) { 
        printf("Step 2: -------------------------------------------------------\n");
        print_result((double *)test_bigArray, 6, 6);
    }
    MPI_Finalize();

    return 0;
}