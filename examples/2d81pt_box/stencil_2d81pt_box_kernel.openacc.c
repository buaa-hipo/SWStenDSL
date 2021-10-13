#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// 4096*4096, halo = 4
#define DIM_1 4096
#define DIM_0 4096
#define HALO 4
#define ITERATION 20

#define M (DIM_1+2*HALO)
#define N (DIM_0+2*HALO)

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

void mpe_verify()
{
    int i, j;
    #pragma acc parallel local(i, j) copyin(input_mpe) copyout(tmp_mpe)
    #pragma acc loop independent
    for (i = HALO; i < M-HALO; i++) {
        #pragma acc loop tile(128)
        for (j = HALO; j < N-HALO; j++) {
            tmp_mpe[i][j] = \
                0.1*input_mpe[i-4][j-4] + 0.2*input_mpe[i-4][j-3] + 0.3*input_mpe[i-4][j-2] + 0.4*input_mpe[i-4][j-1] + 0.5*input_mpe[i-4][j] + 0.6*input_mpe[i-4][j+1] + 0.7*input_mpe[i-4][j+2] + 0.8*input_mpe[i-4][j+3] + 0.9*input_mpe[i-4][j+4] +
                0.1*input_mpe[i-3][j-4] + 0.2*input_mpe[i-3][j-3] + 0.3*input_mpe[i-3][j-2] + 0.4*input_mpe[i-3][j-1] + 0.5*input_mpe[i-3][j] + 0.6*input_mpe[i-3][j+1] + 0.7*input_mpe[i-3][j+2] + 0.8*input_mpe[i-3][j+3] + 0.9*input_mpe[i-3][j+4] +
                0.1*input_mpe[i-2][j-4] + 0.2*input_mpe[i-2][j-3] + 0.3*input_mpe[i-2][j-2] + 0.4*input_mpe[i-2][j-1] + 0.5*input_mpe[i-2][j] + 0.6*input_mpe[i-2][j+1] + 0.7*input_mpe[i-2][j+2] + 0.8*input_mpe[i-2][j+3] + 0.9*input_mpe[i-2][j+4] +
                0.1*input_mpe[i-1][j-4] + 0.2*input_mpe[i-1][j-3] + 0.3*input_mpe[i-1][j-2] + 0.4*input_mpe[i-1][j-1] + 0.5*input_mpe[i-1][j] + 0.6*input_mpe[i-1][j+1] + 0.7*input_mpe[i-1][j+2] + 0.8*input_mpe[i-1][j+3] + 0.9*input_mpe[i-1][j+4] +
                0.1*input_mpe[i][j-4] + 0.2*input_mpe[i][j-3] + 0.3*input_mpe[i][j-2] + 0.4*input_mpe[i][j-1] + 0.5*input_mpe[i][j] + 0.6*input_mpe[i][j+1] + 0.7*input_mpe[i][j+2] + 0.8*input_mpe[i][j+3] + 0.9*input_mpe[i][j+4] +
                0.1*input_mpe[i+1][j-4] + 0.2*input_mpe[i+1][j-3] + 0.3*input_mpe[i+1][j-2] + 0.4*input_mpe[i+1][j-1] + 0.5*input_mpe[i+1][j] + 0.6*input_mpe[i+1][j+1] + 0.7*input_mpe[i+1][j+2] + 0.8*input_mpe[i+1][j+3] + 0.9*input_mpe[i+1][j+4] +
                0.1*input_mpe[i+2][j-4] + 0.2*input_mpe[i+2][j-3] + 0.3*input_mpe[i+2][j-2] + 0.4*input_mpe[i+2][j-1] + 0.5*input_mpe[i+2][j] + 0.6*input_mpe[i+2][j+1] + 0.7*input_mpe[i+2][j+2] + 0.8*input_mpe[i+2][j+3] + 0.9*input_mpe[i+2][j+4] +
                0.1*input_mpe[i+3][j-4] + 0.2*input_mpe[i+3][j-3] + 0.3*input_mpe[i+3][j-2] + 0.4*input_mpe[i+3][j-1] + 0.5*input_mpe[i+3][j] + 0.6*input_mpe[i+3][j+1] + 0.7*input_mpe[i+3][j+2] + 0.8*input_mpe[i+3][j+3] + 0.9*input_mpe[i+3][j+4] +
                0.1*input_mpe[i+4][j-4] + 0.2*input_mpe[i+4][j-3] + 0.3*input_mpe[i+4][j-2] + 0.4*input_mpe[i+4][j-1] + 0.5*input_mpe[i+4][j] + 0.6*input_mpe[i+4][j+1] + 0.7*input_mpe[i+4][j+2] + 0.8*input_mpe[i+4][j+3] + 0.9*input_mpe[i+4][j+4];
        }
    }
}

int main(int argc, char *argv[])
{
    int i, j, iter;

    // 初始化输入
    printf("Initializing data...\n");
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            double value = rand() % 50;
            input_mpe[i][j] = value;
            tmp_mpe[i][j] = value;
        }
    }

    // 检查结果正确性
    printf("mpe computing\n");
    tic();
    for (iter = 0; iter < ITERATION; iter++)
        mpe_verify();
    double elapsedTime = tok();
    printf("MPE elapsed Time: %lf (ms)\n", elapsedTime);

    exit(EXIT_SUCCESS);
}