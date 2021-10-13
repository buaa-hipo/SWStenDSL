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

void stencil_2d81pt_box(double arg0[4104][4104], double arg1[4104][4104]) {
    struct spe_arg param;
    param.arg0=arg0;
    param.arg1=arg1;
    athread_spawn(func0, &param);
    athread_join();

}

void stencil_2d81pt_box_iteration(double arg0[4104][4104], double arg1[4104][4104]) {
    long i;
    for (i = 0; i < 10; i++) {
        stencil_2d81pt_box(arg0, arg1);
        stencil_2d81pt_box(arg1, arg0);
    }
}
