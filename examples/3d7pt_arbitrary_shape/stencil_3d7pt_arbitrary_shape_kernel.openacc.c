#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// 256*256*256
#define DIM_2 256
#define DIM_1 256
#define DIM_0 256
#define ITERATION 20

#define M (DIM_2+2)
#define N (DIM_1+2)
#define Q (DIM_0+2)

double input_mpe[M][N][Q];
double tmp_mpe[M][N][Q];

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
    int i, j, k;

    #pragma acc parallel local(i,j,k) copyin(input_mpe) copyout(tmp_mpe)
    #pragma acc loop independent
    for (i = 0; i < M-2; i++) {
        #pragma acc loop tile(128, 4)
        for (j = 0; j < N-2; j++) {
            for (k = 0; k < Q-2; k++) {
                tmp_mpe[i][j][k] =\
                    0.1*input_mpe[i+2][j][k] + 0.2*input_mpe[i+1][j][k] +
                    0.3*input_mpe[i][j][k] + 0.4*input_mpe[i][j][k+1] + 0.5*input_mpe[i][j][k+2] +
                    0.6*input_mpe[i][j+1][k] + 0.7*input_mpe[i][j+2][k];
            }
        }
    }
}

int main(int argc, char *argv[])
{
    int i, j, k, iter;

    // 初始化输入
    printf("Initializing data...\n");
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < Q; k++) {
                double value = rand() % 50;
                input_mpe[i][j][k] = value;
                tmp_mpe[i][j][k] = value;
            }
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