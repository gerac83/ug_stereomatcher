/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */



#include <assert.h>
#include <helper_cuda.h>
#include "MatchLib_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


////////////////////////////////////////////////////////////////////////////////
// GPU-specific defines
////////////////////////////////////////////////////////////////////////////////
//Maps to a single instruction on G8x / G9x / G10x
#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )

//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}


////////////////////////////////////////////////////////////////////////////////
// Convolution kernel storage
////////////////////////////////////////////////////////////////////////////////
__constant__ float c_Kernel[KERNEL_LENGTH];

extern "C" void setConvolutionKernel(float *h_Kernel)
{
    cudaMemcpyToSymbol(c_Kernel, h_Kernel, KERNEL_LENGTH * sizeof(float));
}


__constant__ float average[KERNEL_LENGTH];
extern "C" void setConvolutionAverageKernel(float *h_average)
{
    cudaMemcpyToSymbol(average, h_average, KERNEL_LENGTH * sizeof(float));
}


///////////////////////////////////////////////////////////////////////////////

texture<float, 2, cudaReadModeElementType> texSrc;
texture<float, 2, cudaReadModeElementType> texdispx;
texture<float, 2, cudaReadModeElementType> texdispy;
texture<float, 2, cudaReadModeElementType> texdispx2;
texture<float, 2, cudaReadModeElementType> texdispy2;


////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
/*#define   ROWS_BLOCKDIM_X 16
#define   ROWS_BLOCKDIM_Y 4
#define ROWS_RESULT_STEPS 8
#define   ROWS_HALO_STEPS 1
*/
__global__ void convolutionRowsKernel(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int pitch
)
{
    __shared__ float s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

    //Offset to the left halo edge
    const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    int Idx,remainX,tempIdx, tempRm;
    remainX = imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X);
    Idx = imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X);
    tempIdx = remainX / ROWS_BLOCKDIM_X;
    tempRm = remainX % ROWS_BLOCKDIM_X;

    //Load main data
#pragma unroll

    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
    }

    //Load left halo
#pragma unroll

    for (int i = 0; i < ROWS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }

    //Load right halo
#pragma unroll

    for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }

    //Compute and store results
    __syncthreads();

    if(blockIdx.x >= Idx && baseY < imageH)
    {
#pragma unroll

	for (int i = ROWS_BLOCKDIM_X * ROWS_HALO_STEPS; i < ROWS_BLOCKDIM_X * (ROWS_HALO_STEPS + tempIdx) + tempRm; i++)
	{
            float sum = 0;

#pragma unroll

	    for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
	    {
		sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.y][threadIdx.x + i + j];
	    }

	    d_Dst[i] = sum;
	}
    }
    else if (baseY < imageH){
#pragma unroll

	for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
	{
	    float sum = 0;

#pragma unroll

	    for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
            {
		sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
            }

	    d_Dst[i * ROWS_BLOCKDIM_X] = sum;
	}
    }
}


extern "C" void convolutionRowsGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH
)
{
    assert(ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS);
//    assert(imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0);
//    assert(imageH % ROWS_BLOCKDIM_Y == 0);

//    dim3 blocks(imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), imageH / ROWS_BLOCKDIM_Y);
    dim3 blocks(iDivUp(imageW, (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X)), iDivUp(imageH, ROWS_BLOCKDIM_Y));
    dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);

    convolutionRowsKernel<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
        imageW
    );
//	checkCudaErrors(cudaDeviceSynchronize());
    getLastCudaError("convolutionRowsKernel() execution failed\n");
}



////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
/*#define   COLUMNS_BLOCKDIM_X 16
#define   COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_RESULT_STEPS 8
#define   COLUMNS_HALO_STEPS 1
*/
__global__ void convolutionColumnsKernel(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int pitch
)
{
    __shared__ float s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    int Idy,remainY,tempIdy, tempRm;
    remainY = (imageH) % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y);
    Idy = (imageH) / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y);
    tempIdy = remainY / COLUMNS_BLOCKDIM_Y;
    tempRm = remainY % COLUMNS_BLOCKDIM_Y;

    //Main data
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];
    }

    //Upper halo
#pragma unroll

    for (int i = 0; i < COLUMNS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
    }

    //Lower halo
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
    }

    //Compute and store results
    __syncthreads();

    if(blockIdx.y >= Idy && baseX < imageW)
    {
#pragma unroll

	for (int i = COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS; i < COLUMNS_BLOCKDIM_Y * (COLUMNS_HALO_STEPS + tempIdy) + tempRm; i++)
	{
            float sum = 0;

#pragma unroll

	    for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
	    {
		sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.x][threadIdx.y + i + j];
	    }

        d_Dst[i * pitch] = sum;
	}
    }
    else if(baseX < imageW){
#pragma unroll

	for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
	{
            float sum = 0;
#pragma unroll

	    for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
	    {
		sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
	    }

	    d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
	}
    }
}


extern "C" void convolutionColumnsGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH
)
{
    assert(COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS);
//    assert(imageW % COLUMNS_BLOCKDIM_X == 0);
//    assert(imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0);

//    dim3 blocks(imageW / COLUMNS_BLOCKDIM_X, imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
    dim3 blocks(iDivUp(imageW, COLUMNS_BLOCKDIM_X), iDivUp(imageH, (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y)));
    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);
//printf("blocksX=%d,blocksY=%d\n",iDivUp(imageW, COLUMNS_BLOCKDIM_X),iDivUp(imageH, (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y)));
    convolutionColumnsKernel<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
        imageW
    );
//	checkCudaErrors(cudaDeviceSynchronize());
    getLastCudaError("convolutionColumnsKernel() execution failed\n");
}


////////////////////////////////////////////////////////////////////////////////
// Texture Subsample
////////////////////////////////////////////////////////////////////////////////
__global__ void subsampleKernel(
    float *d_Dst,
    int imageW,
    int imageH,
    float scalefactor,
    int imageW2,
    int imageH2
)
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if (ix >= imageW2 || iy >= imageH2)
    {
        return;
    }

    if(x * scalefactor <imageW && y * scalefactor<imageH)
    {
	d_Dst[IMAD(iy, imageW2, ix)] = tex2D(texSrc, x * scalefactor , y * scalefactor);
    }
    else
    {
//	d_Dst[IMAD(iy, imageW2, ix)] = tex2D(texSrc, imageW-1 , imageH-1);
return;
    }
}


extern "C" void subsampleGPU(
    float *d_Dst,
    cudaArray *a_Src,
    int imageW,
    int imageH,
    float scalefactor,
    int imageW2,
    int imageH2
)
{
    dim3 threads(16, 12);
    dim3 blocks(iDivUp(imageW2, threads.x), iDivUp(imageH2, threads.y));

    checkCudaErrors(cudaBindTextureToArray(texSrc, a_Src));
    subsampleKernel<<<blocks, threads>>>(
        d_Dst,
        imageW,
        imageH,
	scalefactor,
	imageW2,
	imageH2
    );
    getLastCudaError("convolutionRowsKernel() execution failed\n");

    checkCudaErrors(cudaUnbindTexture(texSrc));
}


////////////////////////////////////////////////////////////////////////////////

__global__ void subsampleDispKernel(
    float *d_Dst,
    int imageW,
    int imageH,
    float scalefactor,
    int imageW2,
    int imageH2
)
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if (ix >= imageW2 || iy >= imageH2)
    {
        return;
    }

//    if(x * scalefactor <imageW || y * scalefactor<imageH)
//    {
	float src = tex2D(texSrc,x * scalefactor,y * scalefactor);
	d_Dst[IMAD(iy, imageW2, ix)] = SCALE * src;
//    }
//    else
//    {
//	d_Dst[IMAD(iy, imageW2, ix)] = SCALE*tex2D(texSrc, imageW-1 , imageH-1);
//return;
//    }
}


extern "C" void subsampleDispGPU(
    float *d_Dst,
    cudaArray *a_Src,
    int imageW,
    int imageH,
    float scalefactor,
    int imageW2,
    int imageH2
)
{
    dim3 threads(16, 12);
    dim3 blocks(iDivUp(imageW2, threads.x), iDivUp(imageH2, threads.y));

    checkCudaErrors(cudaBindTextureToArray(texSrc, a_Src));
    subsampleDispKernel<<<blocks, threads>>>(
        d_Dst,
        imageW,
        imageH,
	scalefactor,
	imageW2,
	imageH2
    );
    getLastCudaError("convolutionRowsKernel() execution failed\n");

    checkCudaErrors(cudaUnbindTexture(texSrc));
}

////////////////////////////////////////////////////////////////////////////////



__global__ void partsubsampleDispKernel(
    float *d_Dst,
    int imageW,
    int imageH,
    float scalefactor,
    int imageW2,
    int imageH2
)
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if (ix >= imageW2 || iy >= imageH2)
    {
        return;
    }

 //   if(x * scalefactor <imageW || y * scalefactor<imageH)
//    {
	float src = tex2D(texSrc,x / scalefactor,y / scalefactor);
	d_Dst[IMAD(iy, imageW2, ix)] = scalefactor * src;
 //   }
//    else
//    {
//	d_Dst[IMAD(iy, imageW2, ix)] = SCALE*tex2D(texSrc, imageW-1 , imageH-1);
//return;
//    }
}


extern "C" void partsubsampleDispGPU(
    float *d_Dst,
    cudaArray *a_Src,
    int imageW,
    int imageH,
    float scalefactor,
    int imageW2,
    int imageH2
)
{
    dim3 threads(16, 12);
    dim3 blocks(iDivUp(imageW2, threads.x), iDivUp(imageH2, threads.y));

    checkCudaErrors(cudaBindTextureToArray(texSrc, a_Src));
    partsubsampleDispKernel<<<blocks, threads>>>(
        d_Dst,
        imageW,
        imageH,
	scalefactor,
	imageW2,
	imageH2
    );
    getLastCudaError("convolutionRowsKernel() execution failed\n");

    checkCudaErrors(cudaUnbindTexture(texSrc));
}



////////////////////////////////////////////////////////////////////////////////
// Warp Right Image
////////////////////////////////////////////////////////////////////////////////
__global__ void warpAbyB(
    float *d_Dst,
    int imageW,
    int imageH
)
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if(ix <imageW && iy <imageH)
    {
	float warpx = tex2D(texdispx,x,y);
	float warpy = tex2D(texdispy,x,y);
	d_Dst[IMAD(iy, imageW, ix)] = tex2D(texSrc, x+warpx, y+warpy);
    }
    else
    {
	return;
    }
}


extern "C" void warp(
    float *d_Dst,
    cudaArray *a_Src,
    cudaArray *dispx,
    cudaArray *dispy,
    int imageW,
    int imageH
)
{
    dim3 threads(16, 12);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));

    checkCudaErrors(cudaBindTextureToArray(texSrc, a_Src));
    checkCudaErrors(cudaBindTextureToArray(texdispx, dispx));
    checkCudaErrors(cudaBindTextureToArray(texdispy, dispy));

    warpAbyB<<<blocks, threads>>>(
        d_Dst,
        imageW,
        imageH
    );
    getLastCudaError("convolutionRowsKernel() execution failed\n");

    checkCudaErrors(cudaUnbindTexture(texSrc));
    checkCudaErrors(cudaUnbindTexture(texdispx));
    checkCudaErrors(cudaUnbindTexture(texdispy));
}



////////////////////////////////////////////////////////////////////////////////
// Compare Images
////////////////////////////////////////////////////////////////////////////////
__global__ void Square(
    float *d_Dst,
    int imageW,
    int imageH
)
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if(ix <imageW && iy <imageH)
    {
	float src = tex2D(texSrc,x,y);
	d_Dst[IMAD(iy, imageW, ix)] = src * src;
//	d_Dst[IMAD(iy, imageW, ix)] = ((src-128)/256) * ((src-128)/256);
//	if(d_Dst[IMAD(iy, imageW, ix)] ==0){d_Dst[IMAD(iy, imageW, ix)] =0.00125;}
    }
    else
    {
	return;
    }
}


extern "C" void compareSquareIm(
    float *d_Dst,
    cudaArray *a_Src,
    int imageW,
    int imageH
)
{
    dim3 threads(16, 12);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));

    checkCudaErrors(cudaBindTextureToArray(texSrc, a_Src));

    Square<<<blocks, threads>>>(
	d_Dst,
        imageW,
        imageH
    );
    getLastCudaError("convolutionRowsKernel() execution failed\n");

    checkCudaErrors(cudaUnbindTexture(texSrc));
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__global__ void CompareMove(
                            float *d_Dst,
                            int imageW,
                            int imageH,
                            float thresholdx,
                            float thresholdy
                            )
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;
    
    if(ix <imageW && iy <imageH)
    {
        float src = tex2D(texSrc, x, y);
        float warpx = tex2D(texdispx, x+thresholdx, y+thresholdy);
        d_Dst[IMAD(iy, imageW, ix)] = src * warpx;
        //	d_Dst[IMAD(iy, imageW, ix)] = ((src-128)/256) * ((warpx-128)/256);
        //	if(d_Dst[IMAD(iy, imageW, ix)] ==0){d_Dst[IMAD(iy, imageW, ix)] =0.00125;}
    }
    else
    {
        return;
    }
}


extern "C" void compareImMove(
                              float *d_Dst,
                              cudaArray *a_Src,
                              cudaArray *dispx,
                              int imageW,
                              int imageH,
                              float thresholdx,
                              float thresholdy
                              )
{
    dim3 threads(16, 12);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));
    
    checkCudaErrors(cudaBindTextureToArray(texSrc, a_Src));
    checkCudaErrors(cudaBindTextureToArray(texdispx, dispx));
    CompareMove<<<blocks, threads>>>(
                                     d_Dst,
                                     imageW,
                                     imageH,
                                     thresholdx,
                                     thresholdy
                                     );
    getLastCudaError("convolutionRowsKernel() execution failed\n");
    
    checkCudaErrors(cudaUnbindTexture(texSrc));
    checkCudaErrors(cudaUnbindTexture(texdispx));
}


////////////////////////////////////////////////////////////////////////////////

__global__ void MoveCorrelation(
                                float *d_Dst,
                                int imageW,
                                int imageH,
                                float thresholdx,
                                float thresholdy
                                )
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;
    
    if(ix <imageW && iy <imageH)
    {
        float src = tex2D(texSrc,x,y);
        float warpl = tex2D(texdispx,x,y);
        float warpr = tex2D(texdispy,x+thresholdx,y+thresholdy);
        d_Dst[IMAD(iy, imageW, ix)] = (src*src) / (warpl * warpr);
        //	d_Dst[IMAD(iy, imageW, ix)] = src / (sqrt(warpl) * sqrt(warpr) +0.0000);
        if(d_Dst[IMAD(iy, imageW, ix)] >1){d_Dst[IMAD(iy, imageW, ix)] =1.0;}
        if(d_Dst[IMAD(iy, imageW, ix)] <0){d_Dst[IMAD(iy, imageW, ix)] =0.0;}
    }
    else
    {
        return;
    }
}


extern "C" void calculateImMoveCorr(
                                    float *d_Dst,
                                    cudaArray *dispx,
                                    cudaArray *dispy,
                                    cudaArray *a_Src,
                                    int imageW,
                                    int imageH,
                                    float thresholdx,
                                    float thresholdy
                                    )
{
    dim3 threads(16, 12);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));
    
    checkCudaErrors(cudaBindTextureToArray(texSrc, a_Src));
    checkCudaErrors(cudaBindTextureToArray(texdispx, dispx));
    checkCudaErrors(cudaBindTextureToArray(texdispy, dispy));
    MoveCorrelation<<<blocks, threads>>>(
                                         d_Dst,
                                         imageW,
                                         imageH,
                                         thresholdx,
                                         thresholdy
                                         );
    getLastCudaError("convolutionRowsKernel() execution failed\n");
    
    checkCudaErrors(cudaUnbindTexture(texSrc));
    checkCudaErrors(cudaUnbindTexture(texdispx));
    checkCudaErrors(cudaUnbindTexture(texdispy));
}

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////

__global__ void calculateMean(
    float *d_Dst,
    int imageW,
    int imageH
)
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if(ix <imageW && iy <imageH)
    {
	float src = tex2D(texSrc,x,y);
	float warpl = tex2D(texdispx,x,y);
	float warpr = tex2D(texdispy,x,y);
	d_Dst[IMAD(iy, imageW, ix)] = (src + warpl + warpr) / 3;
//	if(d_Dst[IMAD(iy, imageW, ix)] ==0){d_Dst[IMAD(iy, imageW, ix)] =0.00125;}
    }
    else
    {
	return;
    }
}


extern "C" void calculateMeanCorr(
    float *d_Dst,
    cudaArray *dispx,
    cudaArray *dispy,
    cudaArray *a_Src,
    int imageW,
    int imageH
)
{
    dim3 threads(16, 12);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));

    checkCudaErrors(cudaBindTextureToArray(texSrc, a_Src));
    checkCudaErrors(cudaBindTextureToArray(texdispx, dispx));
    checkCudaErrors(cudaBindTextureToArray(texdispy, dispy));
    calculateMean<<<blocks, threads>>>(
        d_Dst,
        imageW,
        imageH
    );
    getLastCudaError("convolutionRowsKernel() execution failed\n");

    checkCudaErrors(cudaUnbindTexture(texSrc));
    checkCudaErrors(cudaUnbindTexture(texdispx));
    checkCudaErrors(cudaUnbindTexture(texdispy));
}


//////////////////////////////////////////////////////////////


__global__ void PolyDisparity(
    float *d_hor,
    float *d_corr,
    int imageW,
    int imageH,
    float threshold
)
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if(ix <(imageW) && iy <(imageH))
    {
	float c = tex2D(texSrc,x,y);
	float l = tex2D(texdispx,x,y);
	float r = tex2D(texdispx2,x,y);

	 float b1 = (r-l)/2;
	 float c1 = r - (c+b1);

	if(c1<0) {
		d_hor[IMAD(iy, imageW, ix)]=(-b1 * 0.5)/c1;	
		d_hor[IMAD(iy, imageW, ix)]=min (threshold, (max (d_hor[IMAD(iy, imageW, ix)],(0.0-threshold)))) ;
//		d_corr[IMAD(iy, imageW, ix)]=c+b1*d_hor[IMAD(iy, imageW, ix)]+c1*d_hor[IMAD(iy, imageW, ix)]*d_hor[IMAD(iy, imageW, ix)];

	 float cstar = (c1 * d_hor[IMAD(iy, imageW, ix)] +  b1) * d_hor[IMAD(iy, imageW, ix)] + c;

	    if(cstar > 1.0){
		float d = cstar - c;
		if (d > 1e-10){
		    d_hor[IMAD(iy, imageW, ix)] = d_hor[IMAD(iy, imageW, ix)] * ((1.0 - c) / d);
		}
		d_corr[IMAD(iy, imageW, ix)] = 1.0;
		return;
	    }
	    else{
//		d_hor[IMAD(iy, imageW, ix)] = (r-l)/(r+c+l);
//		cstar = c;
		d_corr[IMAD(iy, imageW, ix)] = 0.3 * cstar + 0.7;
		return;
	    }
	}
	else {
	    d_hor[IMAD(iy, imageW, ix)]=0.0;	d_corr[IMAD(iy, imageW, ix)] = 0.4;
	}

    }
    else
    {
	return;
    }
}


extern "C" void calculatePolyDisparity(
    float *d_hor,
    float *d_corr,
    cudaArray *dispx,
    cudaArray *dispx2,
    cudaArray *d_Src,
    int imageW,
    int imageH,
    float threshold
)
{
    dim3 threads(16, 12);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));

    checkCudaErrors(cudaBindTextureToArray(texSrc, d_Src));
    checkCudaErrors(cudaBindTextureToArray(texdispx, dispx));
    checkCudaErrors(cudaBindTextureToArray(texdispx2, dispx2));


    PolyDisparity<<<blocks, threads>>>(
        d_hor,
        d_corr,
        imageW,
        imageH,
	threshold
    );
    getLastCudaError("convolutionRowsKernel() execution failed\n");

    checkCudaErrors(cudaUnbindTexture(texSrc));
    checkCudaErrors(cudaUnbindTexture(texdispx));
    checkCudaErrors(cudaUnbindTexture(texdispx2));

}

///////////////////////////////////////////////////////////////////////////////



__global__ void compCorrelationKernel(
    float *d_Dst,
    int imageW,
    int imageH
)
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if(ix <(imageW) && iy <(imageH))
    {
	float src = tex2D(texSrc, x, y);
	float warpy = tex2D(texdispy, x, y);
//	d_Dst[IMAD(iy, imageW, ix)] = max(src, warpy);
	d_Dst[IMAD(iy, imageW, ix)] = src * warpy;
    }
//else if((ix >=(imageW-1)&&ix <(imageW)) || (iy >=(imageH-1)&&iy <(imageH))){d_Dst[IMAD(iy, imageW, ix)]=0;}

    else
    {
	return;
    }
}


extern "C" void compCorrelation(
    float *d_Dst,
    cudaArray *dispy,
    cudaArray *a_Src,
    int imageW,
    int imageH
)
{
    dim3 threads(16, 12);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));

    checkCudaErrors(cudaBindTextureToArray(texSrc, a_Src));
    checkCudaErrors(cudaBindTextureToArray(texdispy, dispy));
    compCorrelationKernel<<<blocks, threads>>>(
        d_Dst,
        imageW,
        imageH
    );
    getLastCudaError("convolutionRowsKernel() execution failed\n");

    checkCudaErrors(cudaUnbindTexture(texSrc));
    checkCudaErrors(cudaUnbindTexture(texdispy));
}


////////////////////////////////////////////////////////////////////////////////

__global__ void Disparity(
    float *d_Dst,
    int imageW,
    int imageH
)
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if(ix <(imageW) && iy <(imageH))
    {
	float src = tex2D(texSrc, x, y);
	float warpy = tex2D(texdispy, x, y);
	d_Dst[IMAD(iy, imageW, ix)] = src + warpy;
    }
//else if((ix >=(imageW-1)&&ix <(imageW)) || (iy >=(imageH-1)&&iy <(imageH))){d_Dst[IMAD(iy, imageW, ix)]=0;}

    else
    {
	return;
    }
}


extern "C" void calculateTrueDisparity(
    float *d_Dst,
    cudaArray *dispy,
    cudaArray *a_Src,
    int imageW,
    int imageH
)
{
    dim3 threads(16, 12);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));

    checkCudaErrors(cudaBindTextureToArray(texSrc, a_Src));
    checkCudaErrors(cudaBindTextureToArray(texdispy, dispy));
    Disparity<<<blocks, threads>>>(
        d_Dst,
        imageW,
        imageH
    );
    getLastCudaError("convolutionRowsKernel() execution failed\n");

    checkCudaErrors(cudaUnbindTexture(texSrc));
    checkCudaErrors(cudaUnbindTexture(texdispy));
}

////////////////////////////////////////////////////////////////////////////////

__global__ void TrueConfidence(
    float *d_Dst,
    int imageW,
    int imageH
)
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if(ix <imageW && iy <imageH)
    {
	float src = tex2D(texSrc, x, y);
	float warpy = tex2D(texdispy, x, y);
	d_Dst[IMAD(iy, imageW, ix)] = 0.75*src + 0.25*warpy;
	if(d_Dst[IMAD(iy, imageW, ix)] >1){d_Dst[IMAD(iy, imageW, ix)] =1.0;}
	if(d_Dst[IMAD(iy, imageW, ix)] <0){d_Dst[IMAD(iy, imageW, ix)] =0.0;}
    }
    else
    {
	return;
    }
}


extern "C" void calculateTrueConfidence(
    float *d_Dst,
    cudaArray *dispy,
    cudaArray *a_Src,
    int imageW,
    int imageH
)
{
    dim3 threads(16, 12);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));

    checkCudaErrors(cudaBindTextureToArray(texSrc, a_Src));
    checkCudaErrors(cudaBindTextureToArray(texdispy, dispy));
    TrueConfidence<<<blocks, threads>>>(
        d_Dst,
        imageW,
        imageH
    );
    getLastCudaError("convolutionRowsKernel() execution failed\n");

    checkCudaErrors(cudaUnbindTexture(texSrc));
    checkCudaErrors(cudaUnbindTexture(texdispy));
}

/////////////////////////////////////////////////////

__global__ void scaleDisparityKernel(
    float *d_Dst,
    int m,
    int imageW,
    int imageH
)
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if(ix <imageW && iy <imageH)
    {
	float src = tex2D(texSrc, x, y);
	d_Dst[IMAD(iy, imageW, ix)] = src * m;
    }
    else
    {
	return;
    }
}


extern "C" void scaleDisparity(
    float *d_Dst,
    cudaArray *a_Src,
    int m,
    int imageW,
    int imageH
)
{
    dim3 threads(16, 12);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));

    checkCudaErrors(cudaBindTextureToArray(texSrc, a_Src));
    scaleDisparityKernel<<<blocks, threads>>>(
        d_Dst,
	m,
        imageW,
        imageH
    );
    getLastCudaError("convolutionRowsKernel() execution failed\n");

    checkCudaErrors(cudaUnbindTexture(texSrc));
}


/////////////////////////////////////////////////////

__global__ void smoothKernel(
    float *d_Dst,
    int imageW,
    int imageH
)
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    float sumDisp=0;
    float sumCorr=0;

    if(ix >0 && iy>0 && ix <imageW && iy <imageH)
    {
	float src = tex2D(texSrc, x, y);
	float conf = tex2D(texdispx, x, y);
	sumDisp = src * conf + sumDisp;
	sumCorr = sumCorr + conf;

	src = tex2D(texSrc, x-1, y);
	conf = tex2D(texdispx, x-1, y);
	sumDisp = src * conf + sumDisp;
	sumCorr = sumCorr + conf;

	src = tex2D(texSrc, x+1, y);
	conf = tex2D(texdispx, x+1, y);
	sumDisp = src * conf + sumDisp;
	sumCorr = sumCorr + conf;

	src = tex2D(texSrc, x, y-1);
	conf = tex2D(texdispx, x, y-1);
	sumDisp = src * conf + sumDisp;
	sumCorr = sumCorr + conf;

	src = tex2D(texSrc, x, y+1);
	conf = tex2D(texdispx, x, y+1);
	sumDisp = src * conf + sumDisp;
	sumCorr = sumCorr + conf;

	d_Dst[IMAD(iy, imageW, ix)] = sumDisp / sumCorr;
    }

 /*   if(ix <(imageW-1) && iy <(imageH-1) && ix >1 && iy >1)
    {

	d_Dst[IMAD(iy, imageW, ix)] = tex2D(texSrc, x, y) ;
    }*/
    else
    {
	return;
    }
}


extern "C" void smooth(
    float *d_Dst,
    cudaArray *a_Src,
    cudaArray *dispx,
    int imageW,
    int imageH
)
{
    dim3 threads(16, 12);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));

    checkCudaErrors(cudaBindTextureToArray(texSrc, a_Src));
    checkCudaErrors(cudaBindTextureToArray(texdispx, dispx));
    smoothKernel<<<blocks, threads>>>(
        d_Dst,
        imageW,
        imageH
    );
    getLastCudaError("convolutionRowsKernel() execution failed\n");

    checkCudaErrors(cudaUnbindTexture(texSrc));
    checkCudaErrors(cudaUnbindTexture(texdispx));
}


////////////////////////////////////////////////////////////////////////////////
__global__ void weightedDifferenceGPUKernel(
    float *d_Dst,
    float  *dispx,
    float  *dispy,
    float  *a_Src,
    int imageW,
    int imageH
)
{
    __shared__ float s_data1[12][16];
    __shared__ float s_data2[12][16];
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

//    float absDif;

    if(ix <imageW && iy <imageH)
    {

	s_data1[threadIdx.y][threadIdx.x] = dispx[iy*imageW+ix];
	s_data2[threadIdx.y][threadIdx.x] = dispy[iy*imageW+ix];
	__syncthreads();
	s_data1[threadIdx.y][threadIdx.x] = abs(s_data1[threadIdx.y][threadIdx.x] - s_data2[threadIdx.y][threadIdx.x]);
	s_data2[threadIdx.y][threadIdx.x] = a_Src[iy*imageW+ix];
	__syncthreads();
//	d_Dst[iy*imageW+ix] = s_data1[threadIdx.y][threadIdx.x] * s_data2[threadIdx.y][threadIdx.x];
	dispx[iy*imageW+ix] = s_data1[threadIdx.y][threadIdx.x] * s_data2[threadIdx.y][threadIdx.x];
d_Dst[iy*imageW+ix] = dispx[iy*imageW+ix];


//	absDif = abs(newd-oldd);
//	d_Dst[IMAD(iy, imageW, ix)] = absDif * conf;
    }
    else
    {
	return;
    }
}


extern "C" void weightedDifferenceGPU(
    float *d_Dst,
    float  *dispx,
    float  *dispy,
    float  *a_Src,
    int imageW,
    int imageH
)
{
    dim3 threads(16, 12);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));

    weightedDifferenceGPUKernel<<<blocks, threads>>>(
        d_Dst,
	dispx,
	dispy,
	a_Src,
        imageW,
        imageH
    );
    getLastCudaError("convolutionRowsKernel() execution failed\n");

}


/////////////////////////////////////////////////////

__global__ void reduceGPUKernel(
    float *g_odata,
    float *g_idata,
    int blockSize,
    int n
)
{
    __shared__ float sdata[512];
//    float *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    float mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if ( i + blockSize < n)
            mySum += g_idata[i+blockSize];

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 256];
        }

        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid <  64)
        {
            sdata[tid] = mySum = mySum + sdata[tid +  64];
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile float *smem = sdata;

        if (blockSize >=  64)
        {
            smem[tid] = mySum = mySum + smem[tid + 32];
        }

        if (blockSize >=  32)
        {
            smem[tid] = mySum = mySum + smem[tid + 16];
        }

        if (blockSize >=  16)
        {
            smem[tid] = mySum = mySum + smem[tid +  8];
        }

        if (blockSize >=   8)
        {
            smem[tid] = mySum = mySum + smem[tid +  4];
        }

        if (blockSize >=   4)
        {
            smem[tid] = mySum = mySum + smem[tid +  2];
        }
 
        if (blockSize >=   2)
        {
            smem[tid] = mySum = mySum + smem[tid +  1];
        }
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}


extern "C" void reduceGPU(
    float *d_odata,
    float  *d_idata,
    int blocks,
    int threads,
    int n
)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);


    reduceGPUKernel<<<dimGrid, dimBlock>>>(
        d_odata,
	d_idata,
	blocks,
	n
    );
    getLastCudaError("convolutionRowsKernel() execution failed\n");

}

////////////////////////////////////////////////////////////////////////////////

__global__ void floatrescaleKernel(
    float *d_Dst,
    float m,
    int imageW,
    int imageH
)
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if(ix <imageW && iy <imageH)
    {
	float src = tex2D(texSrc, x, y);
	float rst = tex2D(texdispx, x, y);
	d_Dst[IMAD(iy, imageW, ix)] = (src+rst) / m;
    }
    else
    {
	return;
    }
}


extern "C" void floatrescale(
    float *d_Dst,
    cudaArray *a_Src,
    cudaArray *dispx,
    float m,
    int imageW,
    int imageH
)
{
    dim3 threads(16, 12);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));

    checkCudaErrors(cudaBindTextureToArray(texSrc, a_Src));
    checkCudaErrors(cudaBindTextureToArray(texdispx, dispx));
    
    floatrescaleKernel<<<blocks, threads>>>(
        d_Dst,
        m,
        imageW,
        imageH
    );
    getLastCudaError("convolutionRowsKernel() execution failed\n");

    checkCudaErrors(cudaUnbindTexture(texSrc));
}


////////////////////////////////////////////////////////////////////////////////
// Loop unrolling templates, needed for best performance
////////////////////////////////////////////////////////////////////////////////
template<int i> __device__ float convolutionRowT(float x, float y)
{
    return
        tex2D(texSrc, x + (float)(KERNEL_RADIUS - i), y) * c_Kernel[i]
        + convolutionRowT<i - 1>(x, y);
}

template<> __device__ float convolutionRowT<-1>(float x, float y)
{
    return 0;
}

template<int i> __device__ float convolutionColumnT(float x, float y)
{
    return
        tex2D(texSrc, x, y + (float)(KERNEL_RADIUS - i)) * c_Kernel[i]
        + convolutionColumnT<i - 1>(x, y);
}

template<> __device__ float convolutionColumnT<-1>(float x, float y)
{
    return 0;
}



////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowsKernelT(
    float *d_Dst,
    int imageW,
    int imageH
)
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

/*    if (ix >= imageW || iy >= imageH)
    {
        return;
    }*/
if (ix < imageW && iy < imageH)
{
    float sum = 0;

//#if(UNROLL_INNER)
 //   sum = convolutionRowT<2 *KERNEL_RADIUS>(x, y);
//#else

    for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
    {
        sum += tex2D(texSrc, x + (float)k, y) * c_Kernel[KERNEL_RADIUS - k];
    }

//#endif

    d_Dst[IMAD(iy, imageW, ix)] = sum;
}
else
{
return;
}

}


extern "C" void convolutionRowsGPUT(
    float *d_Dst,
    cudaArray *a_Src,
    int imageW,
    int imageH
)
{
    dim3 threads(16, 12);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));

    checkCudaErrors(cudaBindTextureToArray(texSrc, a_Src));
    convolutionRowsKernelT<<<blocks, threads>>>(
        d_Dst,
        imageW,
        imageH
    );
    getLastCudaError("convolutionRowsKernel() execution failed\n");

    checkCudaErrors(cudaUnbindTexture(texSrc));
}



////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnsKernelT(
    float *d_Dst,
    int imageW,
    int imageH
)
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

/*    if (ix >= imageW || iy >= imageH)
    {
        return;
    }*/

if (ix < imageW && iy < imageH)
{
    float sum = 0;

//#if(UNROLL_INNER)
//    sum = convolutionColumnT<2 *KERNEL_RADIUS>(x, y);
//#else

    for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
    {
        sum += tex2D(texSrc, x, y + (float)k) * c_Kernel[KERNEL_RADIUS - k];
    }

//#endif

    d_Dst[IMAD(iy, imageW, ix)] = sum;
}
else
{
return;
}

}

extern "C" void convolutionColumnsGPUT(
    float *d_Dst,
    cudaArray *a_Src,
    int imageW,
    int imageH
)
{
    dim3 threads(16, 12);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));

    checkCudaErrors(cudaBindTextureToArray(texSrc, a_Src));
    convolutionColumnsKernelT<<<blocks, threads>>>(
        d_Dst,
        imageW,
        imageH
    );
    getLastCudaError("convolutionColumnsKernel() execution failed\n");

    checkCudaErrors(cudaUnbindTexture(texSrc));
}



////////////////////////////////////////////////////////////////////////////////
// Row convolution filter Ta
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowsKernelTa(
    float *d_Dst,
    int imageW,
    int imageH
)
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

/*    if (ix >= imageW || iy >= imageH)
    {
        return;
    }*/
if (ix < imageW && iy < imageH)
{
    float sum = 0;

//#if(UNROLL_INNER)
 //   sum = convolutionRowT<2 *KERNEL_RADIUS>(x, y);
//#else

    for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
    {
        sum += tex2D(texSrc, x + (float)k, y) * average[KERNEL_RADIUS - k];
    }

//#endif

    d_Dst[IMAD(iy, imageW, ix)] = sum;
}
else
{
return;
}

}


extern "C" void convolutionRowsGPUTa(
    float *d_Dst,
    cudaArray *a_Src,
    int imageW,
    int imageH
)
{
    dim3 threads(16, 12);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));

    checkCudaErrors(cudaBindTextureToArray(texSrc, a_Src));
    convolutionRowsKernelTa<<<blocks, threads>>>(
        d_Dst,
        imageW,
        imageH
    );
    getLastCudaError("convolutionRowsKernel() execution failed\n");

    checkCudaErrors(cudaUnbindTexture(texSrc));
}



////////////////////////////////////////////////////////////////////////////////
// Column convolution filter Ta
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnsKernelTa(
    float *d_Dst,
    int imageW,
    int imageH
)
{
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

/*    if (ix >= imageW || iy >= imageH)
    {
        return;
    }*/

if (ix < imageW && iy < imageH)
{
    float sum = 0;

//#if(UNROLL_INNER)
//    sum = convolutionColumnT<2 *KERNEL_RADIUS>(x, y);
//#else

    for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
    {
        sum += tex2D(texSrc, x, y + (float)k) * average[KERNEL_RADIUS - k];
    }

//#endif

    d_Dst[IMAD(iy, imageW, ix)] = sum;
}
else
{
return;
}

}

extern "C" void convolutionColumnsGPUTa(
    float *d_Dst,
    cudaArray *a_Src,
    int imageW,
    int imageH
)
{
    dim3 threads(16, 12);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));

    checkCudaErrors(cudaBindTextureToArray(texSrc, a_Src));
    convolutionColumnsKernelTa<<<blocks, threads>>>(
        d_Dst,
        imageW,
        imageH
    );
    getLastCudaError("convolutionColumnsKernel() execution failed\n");

    checkCudaErrors(cudaUnbindTexture(texSrc));
}
