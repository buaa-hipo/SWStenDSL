#include <stdio.h>
#include <stdlib.h>

#include "../../utils/mpi_lib.h"

// 每个节点有一个4x4的数组, 共有4x4个进程
double test_array[4][4];

void print_result() {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) 
            printf("%lf ", test_array[i][j]);
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

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            test_array[i][j] = my_rank;
        }
    }

    exchange_halo_2D_double((double *)test_array, 4, 4, 4, 4, 1, 1, 1, 1, my_rank);

    // 输出最终结果
    if (my_rank != 0) {
        MPI_Send(test_array, 4*4, MPI_DOUBLE, 0, 0,MPI_COMM_WORLD);
    } else {
        printf("Process %d:\n", my_rank);
        print_result();

        for (int i = 1; i < comm_size; i++) {
            MPI_Recv(test_array, 4*4, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Process %d:\n", i);
            print_result();
        }
    }
    MPI_Finalize();
    return 0;
}