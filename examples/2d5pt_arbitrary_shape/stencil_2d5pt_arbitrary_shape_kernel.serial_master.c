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

void slave_func0(struct spe_arg * arg);

void stencil_2d5pt(double arg0[4098][4098], double arg1[4098][4098]) {
    struct spe_arg param;
    param.arg0=arg0;
    param.arg1=arg1;
    athread_spawn(func0, &param);
    athread_join();

}

void stencil_2d5pt_iteration(double arg0[4098][4098], double arg1[4098][4098]) {
    long i;
    for (i = 0; i < 10; i++) {
        stencil_2d5pt(arg0, arg1);
        stencil_2d5pt(arg1, arg0);
    }
}
