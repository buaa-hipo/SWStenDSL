#include <stdio.h>
#include <stdlib.h>
#include <athread.h>

double input_mpe[72][18][16], input_spe[72][18][16];
double output_mpe[72][18][16], output_spe[72][18][16];

void laplace(double value_arg0[72][18][16], double value_arg1[72][18][16]);
void laplace_iteration(double value_arg0[72][18][16], double value_arg1[72][18][16]);

int main(int argc, char *argv[])
{
    int i, j, k, iter;
    FILE *mpe_result, *spe_result;
    athread_init();

    // 初始化输入
    for (i = 0; i < 72; i++) {
        for (j = 0; j < 18; j++) {
            for (k = 0; k < 16; k++) {
                // input[i][j][k] = i*100*100+j*100+k;
                // input[i][j][k] = i*j*k;
                double value = rand() % 50;
                input_mpe[i][j][k] = value;
                input_spe[i][j][k] = value;
                // 毒化out_mpe和output_spe内存
                output_mpe[i][j][k] = value;
                output_spe[i][j][k] = value;
            }
        }
    }

    // laplace(input, output_spe);
    laplace_iteration(input_spe, output_spe);

    // MPE 版本
    for (iter = 0; iter < 5; iter++) {
        for (i = 1; i < 71; i++) {
            for (j = 1; j < 17; j++) {
                for (k = 0; k < 16; k++) {
                    output_mpe[i][j][k] = 
                        (input_mpe[i-1][j][k] + input_mpe[i+1][j][k] +
                        input_mpe[i][j+1][k] + input_mpe[i][j-1][k]) -
                        4.0 * input_mpe[i][j][k];
                }
            }
        }
        for (i = 1; i < 71; i++) {
            for (j = 1; j < 17; j++) {
                for (k = 0; k < 16; k++) {
                    input_mpe[i][j][k] = 
                        (output_mpe[i-1][j][k] + output_mpe[i+1][j][k] +
                        output_mpe[i][j+1][k] + output_mpe[i][j-1][k]) -
                        4.0 * output_mpe[i][j][k];
                }
            }
        }
    }

    // 检查正确性
    int flag = 1;
    for (i = 1; i < 71; i++) {
        for (j = 1; j < 17; j++) {
            for (k = 0; k < 16; k++) {
                // double value = output_spe[i][j][k] - output_mpe[i][j][k];
                double value = input_spe[i][j][k] - input_mpe[i][j][k];
                if (value > 1e-6 || value < -1e-6)  {
                    flag = 0;
                    goto exit;
                }

            }
        }
    }

exit:
    if (flag)
        printf("Verify Success\n");
    else
        printf("Verify Failed\n");

    // 输出结果到文件
    mpe_result = fopen("mpe_result", "w");
    spe_result = fopen("spe_result", "w");

    for (i = 0; i < 72; i++) {
        for(j = 0; j < 18; j++) {
            for (k = 0; k < 16; k++) {
                fprintf(mpe_result, "%.2lf ", input_mpe[i][j][k]);
                fprintf(spe_result, "%.2lf ", input_spe[i][j][k]);
            }
            fprintf(mpe_result, "\n");
            fprintf(spe_result, "\n");
        }
        fprintf(mpe_result, "\n");
        fprintf(spe_result, "\n");
    }

    fclose(mpe_result);
    fclose(spe_result);

    athread_halt();
    exit(EXIT_SUCCESS);
}
