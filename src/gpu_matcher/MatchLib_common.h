

#ifndef MATCHLIB_COMMON_H
#define MATCHLIB_COMMON_H

//#ifndef CONVOLUTIONTEXTURE_COMMON_H
//#define CONVOLUTIONTEXTURE_COMMON_H


#define PI 3.1415926
#define KERNEL_RADIUS 2 //8
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)
#define MAX_LEVEL 14	//14
#define SIGMA 1.1
#define SCALE 1.41421356	//1.414
#define LAYER 2
#define BIAS 0.00125


#define   ROWS_BLOCKDIM_X 16
#define   ROWS_BLOCKDIM_Y 4
#define ROWS_RESULT_STEPS 8
#define   ROWS_HALO_STEPS 1

#define   COLUMNS_BLOCKDIM_X 16
#define   COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_RESULT_STEPS 8
#define   COLUMNS_HALO_STEPS 1

#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////
// Reference CPU convolution
////////////////////////////////////////////////////////////////////////////////
extern "C" void convolutionRowCPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR
);

extern "C" void convolutionColumnCPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR
);



////////////////////////////////////////////////////////////////////////////////
// GPU convolution
////////////////////////////////////////////////////////////////////////////////
extern "C" void setConvolutionKernel(float *h_Kernel);
extern "C" void setConvolutionAverageKernel(float *h_average);


extern "C" void convolutionRowsGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH
);

extern "C" void convolutionColumnsGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH
);

#endif
