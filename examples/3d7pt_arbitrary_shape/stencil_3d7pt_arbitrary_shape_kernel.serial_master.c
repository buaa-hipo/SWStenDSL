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

void stencil_3d7pt(double arg0[258][258][258], double arg1[258][258][258]) {
    struct spe_arg param;
    param.arg0=arg0;
    param.arg1=arg1;
    athread_spawn(func0, &param);
    athread_join();

}

void stencil_3d7pt_iteration(double arg0[258][258][258], double arg1[258][258][258]) {
    long i;
    for (i = 0; i < 10; i++) {
        stencil_3d7pt(arg0, arg1);
        stencil_3d7pt(arg1, arg0);
    }
}
