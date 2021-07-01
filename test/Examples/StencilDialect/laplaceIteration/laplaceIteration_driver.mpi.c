#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "utils/mpi_io.h"

double input_mpe[212][50][48], input_spe[72][18][16];
double output_mpe[212][50][48], output_spe[72][18][16];

void laplace(double value_arg0[72][18][16], double value_arg1[72][18][16]);
void laplace_iteration(double value_arg0[72][18][16], double value_arg1[72][18][16]);

int main(int argc, char *argv[])
{
    int i, j, k, iter;
    MPI_Init(&argc, &argv);
    athread_init();

    int my_rank;
    int comm_size;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    if (my_rank == 0)
        printf("Initial data ...\n");
    // 大数组初始化
    for (i = 0; i < 212; i++) {
        for (j = 0; j < 50; j++) {
            for (k = 0; k < 48; k++) {
                double value = rand() % 50;
                input_mpe[i][j][k] = value;
                output_mpe[i][j][k] = value;
            }
        }
    }

    // 数据分发
    if (my_rank == 0)
        printf("Sending data ...\n");
    swsten_send_data_3D_double(input_mpe, 212, 50, 48, input_spe, 72, 18, 16,
                                1, 1, 1, 1, 0, 0, 3, 3, 3);
    memcpy(output_spe, input_spe, 72*18*16*sizeof(double));

    if (my_rank == 0)
        printf("Computing ...\n");
    laplace_iteration(input_spe, output_spe);
    
    // 主进程计算
    if (my_rank == 0) {
        for (iter = 0; iter < 5; iter++) {
            for (i = 1; i < 211; i++) {
                for (j = 1; j < 49; j++) {
                    for (k = 0; k < 48; k++) {
                        output_mpe[i][j][k]= 
                            (input_mpe[i-1][j][k] + input_mpe[i+1][j][k] +
                            input_mpe[i][j+1][k] + input_mpe[i][j-1][k]) -
                            4.0 * input_mpe[i][j][k];
                    }
                }
            }
            for (i = 1; i < 211; i++) {
                for (j = 1; j < 49; j++) {
                    for (k = 0; k < 48; k++) {
                        input_mpe[i][j][k] = 
                            (output_mpe[i-1][j][k] + output_mpe[i+1][j][k] +
                            output_mpe[i][j+1][k] + output_mpe[i][j-1][k]) -
                            4.0 * output_mpe[i][j][k];
                    }
                }
            }
        }
    }

    // 数据回收
    if (my_rank == 0)
        printf("Recving data ...\n");
    swsten_recv_data_3D_double(output_mpe, 212, 50, 48, input_spe, 72, 18, 16,
                                1, 1, 1, 1, 0, 0, 3, 3, 3);

    // 主进程结果检查
    if (my_rank == 0) {
        int flag = 0;
        for (i = 0; i < 212; i++) {
            for (j = 0; j < 50; j++) {
                for (k = 0; k < 48; k++) {
                    if (input_mpe[i][j][k] != output_mpe[i][j][k]) {
                        flag = 1;
                        goto exit;
                    }
                }
            }
        }

exit:
        if (flag)
            printf("Verify Failed\n");
        else
            printf("Verify OK\n");
    }

    if (my_rank == 0) {
        printf("Writing result to file ...\n");
        FILE *mpe_result, *mpi_result;
        mpe_result = fopen("mpe_result", "w");
        mpi_result = fopen("mpi_result", "w");
        for (i = 0; i < 212; i++) {
            for(j = 0; j < 50; j++) {
                for (k = 0; k < 48; k++) {
                    fprintf(mpe_result, "%.2lf ", input_mpe[i][j][k]);
                    fprintf(mpi_result, "%.2lf ", output_mpe[i][j][k]);
                }
                fprintf(mpe_result, "\n");
                fprintf(mpi_result, "\n");
            }
            fprintf(mpe_result, "\n");
            fprintf(mpi_result, "\n");
        }
    }

    athread_halt();
    MPI_Finalize();
    return 0;    
}
