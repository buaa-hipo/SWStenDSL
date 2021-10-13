#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// 4096*4096, halo = 1
#define DIM_1 4096
#define DIM_0 4096
#define HALO 1
#define ITERATION 20
#define HALO_2 16

#define M (DIM_1+2*HALO)
#define N (DIM_0+2*HALO)

// 4096*4096, halo = 1
double input_mpe[M][N];
double tmp_mpe[M][N];

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

void stencil_2d5pt_nested_iteration(double value_arg0[M][N], double value_arg1[M][N]);

void mpe_verify1()
{
    int i, j;
    #pragma acc parallel local(i, j) copyin(input_mpe) copyout(tmp_mpe)
    #pragma acc loop independent
    for (i = HALO; i < M-HALO; i++) {
        #pragma acc loop tile(64)
        for (j = HALO; j < N-HALO; j++) {
            tmp_mpe[i][j] = \
                0.2*input_mpe[i-1][j] 
                + 0.4*input_mpe[i][j-1] - 0.5*input_mpe[i][j] + 0.4*input_mpe[i][j+1] 
                + 0.2*input_mpe[i+1][j];
        }
    }
}
void mpe_verify2()
{
    int i, j;
    #pragma acc parallel local(i, j) copyin(tmp_mpe) copyout(input_mpe)
    #pragma acc loop independent
    for (i = HALO+HALO_2; i < M-HALO-HALO_2; i++) {
        #pragma acc loop tile(32)
        for (j = HALO+HALO_2; j < N-HALO-HALO_2; j++) {
            input_mpe[i][j] = \
                0.1*tmp_mpe[i-1][j-1] + 0.3*tmp_mpe[i-1][j+1] 
                - 0.5*tmp_mpe[i][j]
                + 0.3*tmp_mpe[i+1][j-1] + 0.1*tmp_mpe[i+1][j+1];
        }
    }
}

int main(int argc, char *argv[])
{
    int i, j, iter;

    // 初始化输入
    printf("Initializing data...\n");
    for (i = HALO; i < M-HALO; i++) {
        for (j = HALO; j < N-HALO; j++) {
            double value = rand() % 50;
            input_mpe[i][j] = value;
            tmp_mpe[i][j] = value;
        }
    }

    // 检查结果正确性
    printf("mpe computing\n");
    tic();
    for (iter = 0; iter < ITERATION; iter++) {
        mpe_verify1();
        mpe_verify2();
    }

    double elapsedTime = tok();
    printf("MPE elapsed Time: %lf (ms)\n", elapsedTime);

    exit(EXIT_SUCCESS);
}