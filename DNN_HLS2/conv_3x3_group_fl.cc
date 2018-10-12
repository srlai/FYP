

// conv 3x3 for group (depth-wise convolutions)

#include <stdio.h>
#include <math.h>
#include <ap_fixed.h>
#include "hls_stream.h"
#include "net_hls.h"


void load_weights(FIX_WT weight_buf[32],
				  FIX_WT weights[32][3][3],
				  int i, int j)
{
#pragma HLS ARRAY_PARTITION variable=weights dim=1 factor=16

	for(int coo = 0; coo < 16; coo++){
#pragma HLS unroll
		weight_buf[coo] = weights[coo][i][j];
		//printf("weight_buf[%d] = weights[%d][%d][%d] (%f)\n", coo, coo, i, j, weights[coo][i][j] );
	}
}


void CONV1_3x3_group(FIX_FM bottom[16][34][34],
					FIX_FM top[32][34][34],
					FIX_WT weights[32][3][3],int stride)
{

	FIX_WT weight_buf[32];

#pragma HLS ARRAY_PARTITION variable=bottom cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=top cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=weight_buf complete


	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 3; j++){

#pragma HLS dataflow

			load_weights(weight_buf, weights, i, j);

			for(int h = 1; h <= 32; h = h += stride){
				for(int w = 1; w <= 32; w = w += stride){
#pragma HLS pipeline
					for(int co = 0; co < 32; co++){
#pragma HLS unroll
						//top_tmp[co][h][w] += weight_buf[co] * bottom[co][h+i-1][w+j-1];
						top[co][h/stride + 1][w/stride + 1] += weight_buf[co] * bottom[co][h+i-1][w+j-1];
					}
				}
			}
		}
	}

}
