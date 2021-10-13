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
	double cacheRead[42][74];  // cacheRead
	double cacheWrite[32][64]; // cacheWrite
	double(*readArray)[4106] = (double(*)[4106])(arg->arg0);
	double(*writeArray)[4106] = (double(*)[4106])(arg->arg1);

	long my_id = athread_get_id(-1);
	long i_outer, j_outer, i_inner, j_inner;
	for (i_outer = my_id; i_outer < 128; i_outer += 64) {
		for (j_outer = 0; j_outer < 64; j_outer++) {
			DMA_get(readArray[i_outer * 32][j_outer * 64], cacheRead[0][0], 1, 3108*sizeof(double), 4032*sizeof(double), 74*sizeof(double));
			for (i_inner = 0; i_inner < 32; i_inner++) {
				for (j_inner = 0; j_inner < 64; j_inner++) {
					long base_i = 5 + i_inner;
					long base_j = 5 + j_inner;

                    cacheWrite[base_i][base_j] = \
						0.01*cacheRead[base_i-5][base_j-5] + 0.02*cacheRead[base_i-5][base_j-4] + 0.03*cacheRead[base_i-5][base_j-3] + 0.04*cacheRead[base_i-5][base_j-2] + 0.05*cacheRead[base_i-5][base_j-1] + 0.06*cacheRead[base_i-5][base_j] 
						+ 0.05*cacheRead[base_i-5][base_j+1] + 0.04*cacheRead[base_i-5][base_j+2] + 0.03*cacheRead[base_i-5][base_j+3] + 0.02*cacheRead[base_i-5][base_j+4] + 0.01*cacheRead[base_i-5][base_j+5] +

						0.01*cacheRead[base_i-4][base_j-5] + 0.02*cacheRead[base_i-4][base_j-4] + 0.03*cacheRead[base_i-4][base_j-3] + 0.04*cacheRead[base_i-4][base_j-2] + 0.05*cacheRead[base_i-4][base_j-1] + 0.06*cacheRead[base_i-4][base_j] 
						+ 0.05*cacheRead[base_i-4][base_j+1] + 0.04*cacheRead[base_i-4][base_j+2] + 0.03*cacheRead[base_i-4][base_j+3] + 0.02*cacheRead[base_i-4][base_j+4] + 0.01*cacheRead[base_i-4][base_j+5] +

						0.01*cacheRead[base_i-3][base_j-5] + 0.02*cacheRead[base_i-3][base_j-4] + 0.03*cacheRead[base_i-3][base_j-3] + 0.04*cacheRead[base_i-3][base_j-2] + 0.05*cacheRead[base_i-3][base_j-1] + 0.06*cacheRead[base_i-3][base_j] 
						+ 0.05*cacheRead[base_i-3][base_j+1] + 0.04*cacheRead[base_i-3][base_j+2] + 0.03*cacheRead[base_i-3][base_j+3] + 0.02*cacheRead[base_i-3][base_j+4] + 0.01*cacheRead[base_i-3][base_j+5] +

						0.01*cacheRead[base_i-2][base_j-5] + 0.02*cacheRead[base_i-2][base_j-4] + 0.03*cacheRead[base_i-2][base_j-3] + 0.04*cacheRead[base_i-2][base_j-2] + 0.05*cacheRead[base_i-2][base_j-1] + 0.06*cacheRead[base_i-2][base_j] 
						+ 0.05*cacheRead[base_i-2][base_j+1] + 0.04*cacheRead[base_i-2][base_j+2] + 0.03*cacheRead[base_i-2][base_j+3] + 0.02*cacheRead[base_i-2][base_j+4] + 0.01*cacheRead[base_i-2][base_j+5] +
					
						0.01*cacheRead[base_i-1][base_j-5] + 0.02*cacheRead[base_i-1][base_j-4] + 0.03*cacheRead[base_i-1][base_j-3] + 0.04*cacheRead[base_i-1][base_j-2] + 0.05*cacheRead[base_i-1][base_j-1] + 0.06*cacheRead[base_i-1][base_j] 
						+ 0.05*cacheRead[base_i-1][base_j+1] + 0.04*cacheRead[base_i-1][base_j+2] + 0.03*cacheRead[base_i-1][base_j+3] + 0.02*cacheRead[base_i-1][base_j+4] + 0.01*cacheRead[base_i-1][base_j+5] +

						0.01*cacheRead[base_i][base_j-5] + 0.02*cacheRead[base_i][base_j-4] + 0.03*cacheRead[base_i][base_j-3] + 0.04*cacheRead[base_i][base_j-2] + 0.05*cacheRead[base_i][base_j-1] + 0.06*cacheRead[base_i][base_j] 
						+ 0.05*cacheRead[base_i][base_j+1] + 0.04*cacheRead[base_i][base_j+2] + 0.03*cacheRead[base_i][base_j+3] + 0.02*cacheRead[base_i][base_j+4] + 0.01*cacheRead[base_i][base_j+5] +

						0.01*cacheRead[base_i+1][base_j-5] + 0.02*cacheRead[base_i+1][base_j-4] + 0.03*cacheRead[base_i+1][base_j-3] + 0.04*cacheRead[base_i+1][base_j-2] + 0.05*cacheRead[base_i+1][base_j-1] + 0.06*cacheRead[base_i+1][base_j] 
						+ 0.05*cacheRead[base_i+1][base_j+1] + 0.04*cacheRead[base_i+1][base_j+2] + 0.03*cacheRead[base_i+1][base_j+3] + 0.02*cacheRead[base_i+1][base_j+4] + 0.01*cacheRead[base_i+1][base_j+5] +

						0.01*cacheRead[base_i+2][base_j-5] + 0.02*cacheRead[base_i+2][base_j-4] + 0.03*cacheRead[base_i+2][base_j-3] + 0.04*cacheRead[base_i+2][base_j-2] + 0.05*cacheRead[base_i+2][base_j-1] + 0.06*cacheRead[base_i+2][base_j] 
						+ 0.05*cacheRead[base_i+2][base_j+1] + 0.04*cacheRead[base_i+2][base_j+2] + 0.03*cacheRead[base_i+2][base_j+3] + 0.02*cacheRead[base_i+2][base_j+4] + 0.01*cacheRead[base_i+2][base_j+5] +

						0.01*cacheRead[base_i+3][base_j-5] + 0.02*cacheRead[base_i+3][base_j-4] + 0.03*cacheRead[base_i+3][base_j-3] + 0.04*cacheRead[base_i+3][base_j-2] + 0.05*cacheRead[base_i+3][base_j-1] + 0.06*cacheRead[base_i+3][base_j] 
						+ 0.05*cacheRead[base_i+3][base_j+1] + 0.04*cacheRead[base_i+3][base_j+2] + 0.03*cacheRead[base_i+3][base_j+3] + 0.02*cacheRead[base_i+3][base_j+4] + 0.01*cacheRead[base_i+3][base_j+5] +

						0.01*cacheRead[base_i+4][base_j-5] + 0.02*cacheRead[base_i+4][base_j-4] + 0.03*cacheRead[base_i+4][base_j-3] + 0.04*cacheRead[base_i+4][base_j-2] + 0.05*cacheRead[base_i+4][base_j-1] + 0.06*cacheRead[base_i+4][base_j] 
						+ 0.05*cacheRead[base_i+4][base_j+1] + 0.04*cacheRead[base_i+4][base_j+2] + 0.03*cacheRead[base_i+4][base_j+3] + 0.02*cacheRead[base_i+4][base_j+4] + 0.01*cacheRead[base_i+4][base_j+5] +

						0.01*cacheRead[base_i+5][base_j-5] + 0.02*cacheRead[base_i+5][base_j-4] + 0.03*cacheRead[base_i+5][base_j-3] + 0.04*cacheRead[base_i+5][base_j-2] + 0.05*cacheRead[base_i+5][base_j-1] + 0.06*cacheRead[base_i+5][base_j] 
						+ 0.05*cacheRead[base_i+5][base_j+1] + 0.04*cacheRead[base_i+5][base_j+2] + 0.03*cacheRead[base_i+5][base_j+3] + 0.02*cacheRead[base_i+5][base_j+4] + 0.01*cacheRead[base_i+5][base_j+5];
				}
			}
			long writeArrayIndex_i = i_outer * 32 + 5;
			long writeArrayInde_j = j_outer * 64 + 5;
			DMA_put(cacheWrite[0][0], writeArray[writeArrayIndex_i][writeArrayInde_j], 1, 2048*sizeof(double), 4042*sizeof(double), 64*sizeof(double));
		}
	}
}
