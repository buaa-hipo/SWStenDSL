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
					long base_i = 2 + i_inner;
					long base_j = j_inner;

                    cacheWrite[base_i][base_j] = \
						0.1*cacheWrite[base_i-2][base_j] + 0.2*cacheWrite[base_i-1][base_j] + 
						0.3*cacheWrite[base_i][base_j] + 0.4*cacheWrite[base_i][base_j+1] + 0.5*cacheWrite[base_i][base_j+2];
				}
			}
			long writeArrayIndex_i = i_outer * 32 + 2;
			long writeArrayInde_j = j_outer * 64;
			DMA_put(cacheWrite[0][0], writeArray[writeArrayIndex_i][writeArrayInde_j], 1, 2048*sizeof(double), 4034*sizeof(double), 64*sizeof(double));
		}
	}
}
