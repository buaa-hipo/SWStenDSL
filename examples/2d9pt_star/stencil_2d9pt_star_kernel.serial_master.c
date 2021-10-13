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

void stencil_2d9pt_star(double arg0[4100][4100], double arg1[4100][4100]) {
    struct spe_arg param;
    param.arg0=arg0;
    param.arg1=arg1;
    athread_spawn(func0, &param);
    athread_join();

}

void stencil_2d9pt_star_iteration(double arg0[4100][4100], double arg1[4100][4100]) {
    long i;
    for (i = 0; i < 10; i++) {
        stencil_2d9pt_star(arg0, arg1);
        stencil_2d9pt_star(arg1, arg0);
    }
}
