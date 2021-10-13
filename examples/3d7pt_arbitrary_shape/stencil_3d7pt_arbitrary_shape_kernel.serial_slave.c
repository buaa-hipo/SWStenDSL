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
	double cacheRead[4][10][66];  // cacheRead
	double cacheWrite[2][8][64]; // cacheWrite
	double(*readArray)[258][258] = (double(*)[258][258])(arg->arg0);
	double(*writeArray)[258][258] = (double(*)[258][258])(arg->arg1);

	long my_id = athread_get_id(-1);
	long i_outer, j_outer, k_outer, i_inner, j_inner, k_inner;
	for (i_outer = my_id; i_outer < 128; i_outer += 64) {
		for (j_outer = 0; j_outer < 32; j_outer++) {
			for (k_outer = 0; k_outer < 4; k_outer++) {
				DMA_get(readArray[i_outer * 2+z_iter][j_outer * 8][k_outer * 64], cacheRead[z_iter][0][0], 4, 660*sizeof(double), 192*sizeof(double), 66*sizeof(double));
				for (i_inner = 0; i_inner < 2; i_inner++) {
					for (j_inner = 0; j_inner < 8; j_inner++) {
						for (k_inner = 0; k_inner < 64; k_inner++) {
							long base_i = i_inner;
							long base_j = j_inner;
							long base_k = k_inner;

							cacheWrite[base_i][base_j][base_k] = \
								0.1*cacheRead[base_i+2][base_j][base_k] + 0.2*cacheRead[base_i+1][base_j][base_k] +
								0.3*cacheRead[base_i][base_j][base_k] + 0.4*cacheRead[base_i][base_j][base_k+1] + 0.5*cacheRead[base_i][base_j][base_k+2] +
								0.6*cacheRead[base_i][base_j+1][base_k] + 0.7*cacheRead[base_i][base_j+2][base_k];
						}
					}
				}
				long writeArrayIndex_i = i_outer * 2;
				long writeArrayIndex_j = j_outer * 8;
				long writeArrayIndex_k = k_outer * 64;
				DMA_put(cacheWrite[z_iter][0][0], writeArray[writeArrayIndex_i+z_iter][writeArrayIndex_j][writeArrayIndex_k], 2, 512*sizeof(double), 194*sizeof(double), 64*sizeof(double));
			}
		}
	}
}
