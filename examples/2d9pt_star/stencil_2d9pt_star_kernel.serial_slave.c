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
	double cacheRead[36][68];  // cacheRead
	double cacheWrite[32][64]; // cacheWrite
	double(*readArray)[4100] = (double(*)[4100])(arg->arg0);
	double(*writeArray)[4100] = (double(*)[4100])(arg->arg1);

	long my_id = athread_get_id(-1);
	long i_outer, j_outer, i_inner, j_inner;
	for (i_outer = my_id; i_outer < 128; i_outer += 64) {
		for (j_outer = 0; j_outer < 64; j_outer++) {
			DMA_get(readArray[i_outer * 32][j_outer * 64], cacheRead[0][0], 1, 2448 * sizeof(double), 4032 * sizeof(double), 68 * sizeof(double));
			for (i_inner = 0; i_inner < 32; i_inner++) {
				for (j_inner = 0; j_inner < 64; j_inner++) {
					long base_i = 2 + i_inner;
					long base_j = 2 + j_inner;

                    cacheWrite[base_i][base_j] = \
						  0.1*cacheRead[base_i-2][base_j] + 0.2*cacheRead[base_i-1][base_j] + 0.3*cacheRead[base_i][base_j-2]
                    	+ 0.4*cacheRead[base_i][base_j-1] + 0.5*cacheRead[base_i][base_j] + 0.6*cacheRead[base_i][base_j+1]
                    	+ 0.7*cacheRead[base_i][base_j+2] + 0.8*cacheRead[base_i+1][base_j] + 0.9*cacheRead[base_i+2][base_j];
				}
			}
			long writeArrayIndex_i = i_outer * 32 + 2;
			long writeArrayInde_j = j_outer * 64 + 2;
			DMA_put(cacheWrite[0][0], writeArray[writeArrayIndex_i][writeArrayInde_j], 1, 2048 * sizeof(double), 4036 * sizeof(double), 64 * sizeof(double));
		}
	}
}
