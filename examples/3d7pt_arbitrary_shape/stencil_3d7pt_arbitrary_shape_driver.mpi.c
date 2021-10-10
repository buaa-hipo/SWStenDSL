#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <sys/time.h>

#include "utils/mpi_io.h"

// 256*256*256
#define DIM_2 256
#define DIM_1 256
#define DIM_0 256
#define ITERATION 20

#define M (DIM_2+2)
#define N (DIM_1+2)
#define Q (DIM_0+2)

double input_spe[M][N][Q];
double tmp_spe[M][N][Q];

// 计时辅助函数
struct timeval begin, end;

void tic()
{
    gettimeofday(&begin, NULL);
}

// 返回tic到tok之间的毫秒数
double tok()
{
    gettimeofday(&end, NULL);
    double elapsedTime = (end.tv_sec - begin.tv_sec)*1e3 + \
            (end.tv_usec - begin.tv_usec)*1e-3;
    return elapsedTime;
}

void stencil_3d7pt_iteration(double value_arg0[M][N][Q], double value_arg1[M][N][Q]);

int main(int argc, char *argv[])
{
    int i, j, k;
    MPI_Init(&argc, &argv);
    athread_init();

    int my_rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // 加载数据
    swsten_load_data_from_file_double("matrix", input_spe, M*N*Q);
    memcpy(tmp_spe, input_spe, M*N*Q*sizeof(double));

    // 开始计算
    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0) {
        tic();
        printf("computing...\n");
    }
    stencil_3d7pt_iteration(input_spe, tmp_spe);

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0) {
        double elapsedTime = tok();
        printf("MPI elapsed Time: %lf (ms)\n", elapsedTime);
    }

    swsten_store_data_to_file_double("matrix_result", input_spe, M*N*Q, Q);

    athread_halt();
    MPI_Finalize();
    exit(EXIT_SUCCESS);
}