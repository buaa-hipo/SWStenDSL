#include <athread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <stdint.h>

struct spe_arg {
	double *arg0;
	double *arg1;
};

void slave_func0(struct spe_arg *arg);
void slave_func1(struct spe_arg *arg);

void stencil_2d5pt_nested(double arg0[4098][4098], double arg1[4098][4098]) {
    double *tmp = calloc(16793604, sizeof(double));
    struct spe_arg param;
    param.arg0=arg0;
    param.arg1=tmp;
    athread_spawn(func0, &param);
    athread_join();

    param.arg0 = tmp;
    param.arg1 = arg1;
    athread_spawn(func1, &param);
    athread_join();

    free(tmp);
}

void stencil_2d5pt_nested_iteration(double arg0[4098][4098], double arg1[4098][4098]) {
    long i;
    for (i = 0; i < 10; i++) {
        stencil_2d5pt_nested(arg0, arg1);
        stencil_2d5pt_nested(arg1, arg0);
    }
}
