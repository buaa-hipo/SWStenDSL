#include <slave.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include "utils/dma_lib.h"

struct spe_arg
{
	double *arg0;
	double *arg1;
};

void func0(struct spe_arg *arg)
{
	double cacheRead[34][66];  // cacheRead
	double cacheWrite[32][64]; // cacheWrite
	double(*readArray)[4098] = (double(*)[4098])(arg->arg0);
	double(*writeArray)[4098] = (double(*)[4098])(arg->arg1);

	long my_id = athread_get_id(-1);
	long i_outer, j_outer, i_inner, j_inner;
	for (i_outer = my_id; i_outer < 128; i_outer += 64) {
		for (j_outer = 0; j_outer < 64; j_outer++) {
			DMA_get(readArray[i_outer * 32][j_outer * 64], cacheRead[0][0], 1, 2244*sizeof(double), 4032*sizeof(double), 66*sizeof(double));
			for (i_inner = 0; i_inner < 32; i_inner++) {
				for (j_inner = 0; j_inner < 64; j_inner++) {
					long base_i = 1 + i_inner;
					long base_j = 1 + j_inner;

                    cacheWrite[base_i][base_j] = \
						0.2*cacheRead[base_i-1][base_j] 
						+ 0.4*cacheRead[base_i][base_j-1] - 0.5*cacheRead[base_i][base_j] + 0.4*cacheRead[base_i][base_j+1] 
						+ 0.2*cacheRead[base_i+1][base_j];
				}
			}
			long writeArrayIndex_i = i_outer * 32 + 1;
			long writeArrayInde_j = j_outer * 64 + 1;
			DMA_put(cacheWrite[0][0], writeArray[writeArrayIndex_i][writeArrayInde_j], 1, 2048*sizeof(double), 4034*sizeof(double), 64*sizeof(double));
		}
	}
}

void func1(struct spe_arg *arg)
{
	double cacheRead[50][66];  // cacheRead
	double cacheWrite[16][32]; // cacheWrite
	double(*readArray)[4098] = (double(*)[4098])(arg->arg0);
	double(*writeArray)[4098] = (double(*)[4098])(arg->arg1);

	long my_id = athread_get_id(-1);
	long i_outer, j_outer, i_inner, j_inner;
	for (i_outer = my_id; i_outer < 254; i_outer += 64) {
		for (j_outer = 0; j_outer < 127; j_outer++) {
			DMA_get(readArray[i_outer * 16][j_outer * 32], cacheRead[0][0], 1, 3300*sizeof(double), 4032*sizeof(double), 66*sizeof(double));
			for (i_inner = 0; i_inner < 16; i_inner++) {
				for (j_inner = 0; j_inner < 32; j_inner++) {
					long base_i = 17 + i_inner;
					long base_j = 17 + j_inner;

                    cacheWrite[base_i][base_j] = \
						0.1*cacheRead[base_i-1][base_j-1] + 0.3*cacheRead[base_i-1][base_j+1] 
						- 0.5*cacheRead[base_i][base_j]
						+ 0.3*cacheRead[base_i+1][base_j-1] + 0.1*cacheRead[base_i+1][base_j+1];
				}
			}
			long writeArrayIndex_i = i_outer * 16 + 17;
			long writeArrayInde_j = j_outer * 32 + 17;
			DMA_put(cacheWrite[0][0], writeArray[writeArrayIndex_i][writeArrayInde_j], 1, 512*sizeof(double), 4066*sizeof(double), 32*sizeof(double));
		}
	}
}