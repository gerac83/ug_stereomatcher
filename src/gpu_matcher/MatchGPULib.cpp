// OpenCV

#include "MatchGPULib.h"
#include<math.h>
#include<stdlib.h>
#include<stdio.h>
#include <cv.h>

#include <highgui.h>

// CUDA runtime
#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>

#include "MatchLib_common.h"

#include <stdint.h>     /* unit64_t*/
#include <iostream>
#include <fstream>
#define BILLION 1000000000L

using namespace std;

typedef struct Point {
    float x;
    float y;
} Point;

bool usingMoreGPUMemory = 1;
const int  precision = 5; // the number of simples taken for each weight over the interval
const int  levelcutoff = 22;
const int  smoothtime = 5;
double excutionTime[20];
bool useGPU = 1;
size_t freeMem, totalMem;
////////////////////////////////////////////////////////////////////////////////
// Reference CPU convolution
////////////////////////////////////////////////////////////////////////////////
extern "C" void convolutionRowCPU(
    float *h_Result,
    float *h_Data,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR
);

extern "C" void convolutionColumnCPU(
    float *h_Result,
    float *h_Data,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR
);

////////////////////////////////////////////////////////////////////////////////
// Reference GPU convolution
////////////////////////////////////////////////////////////////////////////////

extern "C" void subsampleGPU(
    float *d_Dst,
    cudaArray *a_Src,
    int imageW,
    int imageH,
    float scalefactor,
    int imageW2,
    int imageH2
);

extern "C" void subsampleDispGPU(
    float *d_Dst,
    cudaArray *a_Src,
    int imageW,
    int imageH,
    float scalefactor,
    int imageW2,
    int imageH2
);

extern "C" void partsubsampleDispGPU(
    float *d_Dst,
    cudaArray *a_Src,
    int imageW,
    int imageH,
    float scalefactor,
    int imageW2,
    int imageH2
);

extern "C" void warp(
    float *d_Dst,
    cudaArray *a_Src,
    cudaArray *dispx,
    cudaArray *dispy,
    int imageW,
    int imageH
);

extern "C" void compareSquareIm(
    float *d_Dst,
    cudaArray *a_Src,
    int imageW,
    int imageH
);

extern "C" void compareImMove(
                              float *d_Dst,
                              cudaArray *a_Src,
                              cudaArray *dispx,
                              int imageW,
                              int imageH,
                              float thresholdx,
                              float thresholdy
                              );

extern "C" void calculateImMoveCorr(
                                    float *d_Dst,
                                    cudaArray *dispx,
                                    cudaArray *dispy,
                                    cudaArray *a_Src,
                                    int imageW,
                                    int imageH,
                                    float thresholdx,
                                    float thresholdy
                                    );


extern "C" void calculateMeanCorr(
    float *d_Dst,
    cudaArray *dispx,
    cudaArray *dispy,
    cudaArray *a_Src,
    int imageW,
    int imageH
);

extern "C" void calculatePolyDisparity(
    float *d_hor,
    float *d_corr,
    cudaArray *dispx,
    cudaArray *dispx2,
    cudaArray *d_Src,
    int imageW,
    int imageH,
    float threshold
);

extern "C" void compCorrelation(
    float *d_Dst,
    cudaArray *dispy,
    cudaArray *a_Src,
    int imageW,
    int imageH
);

extern "C" void calculateTrueDisparity(
    float *d_Dst,
    cudaArray *dispy,
    cudaArray *a_Src,
    int imageW,
    int imageH
);

extern "C" void calculateTrueConfidence(
    float *d_Dst,
    cudaArray *dispy,
    cudaArray *a_Src,
    int imageW,
    int imageH
);

extern "C" void scaleDisparity(
    float *d_Dst,
    cudaArray *a_Src,
    int m,
    int imageW,
    int imageH
);

extern "C" void smooth(
    float *d_Dst,
    cudaArray *a_Src,
    cudaArray *dispx,
    int imageW,
    int imageH
);

extern "C" void weightedDifferenceGPU(
    float *d_Dst,
    float  *dispx,
    float  *dispy,
    float  *a_Src,
    int imageW,
    int imageH
);

extern "C" void reduceGPU(
    float *d_odata,
    float  *d_idata,
    int blocks,
    int threads,
    int n
);

extern "C" void floatrescale(
    float *d_Dst,
    cudaArray *a_Src,
    cudaArray *dispx,
    float m,
    int imageW,
    int imageH
);

extern "C" void convolutionRowsGPUT(
    float *d_Dst,
    cudaArray *a_Src,
    int imageW,
    int imageH
);

extern "C" void convolutionColumnsGPUT(
    float *d_Dst,
    cudaArray *a_Src,
    int imageW,
    int imageH
);

extern "C" void convolutionRowsGPUTa(
    float *d_Dst,
    cudaArray *a_Src,
    int imageW,
    int imageH
);

extern "C" void convolutionColumnsGPUTa(
    float *d_Dst,
    cudaArray *a_Src,
    int imageW,
    int imageH
);

////////////////////////////////////////////////////////////////////////////////

MatchGPULib::MatchGPULib(int argc, char **argv) 
{
	//Use command-line specified CUDA device, otherwise use device with highest Gflops/s
	findCudaDevice(argc, (const char **)argv);
	foveatedmatching = 0; /* The flag will be set later on according to the matcher */
	fovH = 0;
	fovW = 0;
	
	if (argc > 2){
		foveatelevel = atoi(argv[2]);
	}else{
		/*unless otherwise defined on command line*/
		foveatelevel=7;
	}
}


int MatchGPULib::getFoveaWidth(){
	return fovW;
}

int MatchGPULib::getFoveaHeight(){
	return fovH;
}

int MatchGPULib::getFoveateLevel(){
	return foveatelevel;
}

int MatchGPULib::iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}



/* Mozhgan - aug 2014 */
void MatchGPULib::setFoveaWidth(int rows ){
	fovW=rows;
}

void MatchGPULib::setFoveaHeight(int cols){
	fovH=cols;
}
/**/

/* sets the foveated flag passed from RHGPU_matcher*/
void MatchGPULib::setFoveated(int fov)
{
	foveatedmatching = fov;
}

float **MatchGPULib::match(cv_bridge::CvImagePtr cv_ptrL, cv_bridge::CvImagePtr cv_ptrR, int fov){

	int height ,width ,step ,channels;
    uchar *data, *data2;
    float *h_Kernel, *h_average;
    float **image, **image2,**finDisp;
    float ***left, ***right;
    float ***disparity;
    float ***leftFoveated, ***rightFoveated;

    h_Kernel    = (float *)malloc(KERNEL_LENGTH * sizeof(float));
    h_average    = (float *)malloc(KERNEL_LENGTH * sizeof(float));
    height = (int) cv_ptrL->image.rows; /* Tian uses rows as width */
    width = (int) cv_ptrL->image.cols; /**/
    channels = 3;
    step=cv_ptrL->image.step;
    foveatedmatching = fov;
	data = (uchar *) malloc(height * width * channels *sizeof(uchar));
	data = (uchar *) cv_ptrL->image.data;
	data2 = (uchar *) malloc(height * width * channels *sizeof(uchar));
	data2 = (uchar *) cv_ptrR->image.data;
	
	image = (float**)malloc(channels * sizeof(float*));
	image2 = (float**)malloc(channels * sizeof(float*));
    for(int i=0; i<channels; i++){
		image[i] = (float*)malloc(height * width * sizeof(float));
		image2[i] = (float*)malloc(height * width * sizeof(float));
    }
    
    for(int k=0;k<channels;k++)
		for(int i=0;i<height;i++)
			for(int j=0;j<width;j++)
			{
				*(image[k]+i*width+j) = (float)*(data+i*step+j*channels+k);
				*(image2[k]+i*width+j) = (float)*(data2+i*step+j*channels+k);
			}
			
	//  1D gaussian kernel function
    this->gaussiankernel(h_Kernel);
	
    setConvolutionKernel(h_Kernel);
	*h_average=0.0;
	*(h_average+1)=0.3333;
	*(h_average+2)=0.3333;
	*(h_average+3)=0.3333;
	*(h_average+4)=0.0;

    setConvolutionAverageKernel(h_average);

	//Building a pyramid 
    left = CreatePyramidFromImage(image, channels, height, width, h_Kernel);
    right = CreatePyramidFromImage(image2, channels, height, width, h_Kernel);
    if(fov==1){
		leftFoveated = CreateFoveatedPyramid(left, channels, height, width);
		rightFoveated = CreateFoveatedPyramid(right, channels, height, width);
		//Foveated Stereo Matching Part
		disparity = matching(leftFoveated, rightFoveated, channels, height, width);
		finDisp=hierarchicalDisparity(right[0], disparity, channels, width, height);
    }else{
		//Stereo Matching Part
		disparity = matching(left, right, channels, height, width);

		finDisp = (float**)malloc(channels * sizeof(float*));
		for(int i=0; i<channels; i++){
			finDisp[i] = (float*)malloc(height * width * sizeof(float));
		}
		//bottom level of the pyramid
		finDisp = disparity[0];
    } 
    
    free(h_Kernel);
    free(h_average);
/*
    for (int j=0; j<channels; j++) {
        free(image[j]);
        free(image2[j]);
    }
    free(image);
    free(image2);
*/
    for(int i=0; i<MAX_LEVEL; i++){
	for(int k=0;k<channels;k++)
	    {
		free(left[i][k]);
		free(right[i][k]);
		
	    }
	free(left[i]);
	free(right[i]);
	
    }
    free(left);
    free(right);
 
   
    cudaDeviceReset();
    
    return finDisp;
}

/* This function sets the values for the Foveated size */
int MatchGPULib::initStack(cv_bridge::CvImagePtr cv_ptrL, cv_bridge::CvImagePtr cv_ptrR){
int height = (int) cv_ptrL->image.rows; /* Tian uses rows as width */
int width = (int) cv_ptrL->image.cols; /**/
int heighta[MAX_LEVEL-1],widtha[MAX_LEVEL-1];
widtha[0]=width;
heighta[0]=height;

for(int i=0; i<foveatelevel; i++)
    {
	widtha[i+1]=widtha[i]/SCALE;
	heighta[i+1]=heighta[i]/SCALE;
    }

int rows=widtha[foveatelevel-1];
int cols=heighta[foveatelevel-1];

setFoveaHeight(cols);
setFoveaWidth(rows);

return 0;
  }

/* No longer in use for foveated stack retreival - we use 'matchStackPyramid' instead */ 
float ***MatchGPULib::matchStack(cv_bridge::CvImagePtr cv_ptrL, cv_bridge::CvImagePtr cv_ptrR){

	int height ,width ,step ,channels;
    uchar *data, *data2;
    float *h_Kernel, *h_average;
    float **image, **image2,**finDisp;
    float ***left, ***right;
    float ***disparity;
    float ***leftFoveated, ***rightFoveated;

    h_Kernel    = (float *)malloc(KERNEL_LENGTH * sizeof(float));
    h_average    = (float *)malloc(KERNEL_LENGTH * sizeof(float));
    height = (int) cv_ptrL->image.rows; /* Tian uses rows as width */
    width = (int) cv_ptrL->image.cols; /**/

    channels = 3;
    step=cv_ptrL->image.step;
    
	data = (uchar *) malloc(height * width * channels *sizeof(uchar));
	data = (uchar *) cv_ptrL->image.data;
	data2 = (uchar *) malloc(height * width * channels *sizeof(uchar));
	data2 = (uchar *) cv_ptrR->image.data;
	
	image = (float**)malloc(channels * sizeof(float*));
	image2 = (float**)malloc(channels * sizeof(float*));
    for(int i=0; i<channels; i++){
		image[i] = (float*)malloc(height * width * sizeof(float));
		image2[i] = (float*)malloc(height * width * sizeof(float));
    }
    
    for(int k=0;k<channels;k++)
		for(int i=0;i<height;i++)
			for(int j=0;j<width;j++)
			{
				*(image[k]+i*width+j) = (float)*(data+i*step+j*channels+k);
				*(image2[k]+i*width+j) = (float)*(data2+i*step+j*channels+k);
			}
			
	//  1D gaussian kernel function
    this->gaussiankernel(h_Kernel);
	
    setConvolutionKernel(h_Kernel);
	*h_average=0.0;
	*(h_average+1)=0.3333;
	*(h_average+2)=0.3333;
	*(h_average+3)=0.3333;
	*(h_average+4)=0.0;

    setConvolutionAverageKernel(h_average);

	//Building a pyramid 
    left = CreatePyramidFromImage(image, channels, height, width, h_Kernel);
    right = CreatePyramidFromImage(image2, channels, height, width, h_Kernel);
    
	leftFoveated = CreateFoveatedPyramid(left, channels, height, width);
	rightFoveated = CreateFoveatedPyramid(right, channels, height, width);



	//Foveated Stereo Matching Part
	disparity = matching(leftFoveated, rightFoveated, channels, height, width);
	//finDisp=hierarchicalDisparity(right[0], disparity, channels, width, height);

/* debug : print the foveated images
IplImage *outImg; char string[20];
    for(int level=0;level<foveatelevel;level++)
    {
	outImg = cvCreateImage(cvSize(getFoveaWidth(),getFoveaHeight()), 8, 3);  

	for(int k=0;k<channels;k++)
	    for(int l=0;l<outImg->height;l++)
		for(int m=0;m<outImg->width;m++)
		{
		    *(outImg->imageData+l*outImg->widthStep+m*channels+k) = (uchar)*(disparity[level][k]+l*outImg->width+m);

	        }
	sprintf(string,"foveatedDisp%d.bmp",level+1);
	cvSaveImage(string,outImg); 
}
cvReleaseImage(&outImg); // valgrind
*/    

    free(h_Kernel);
    free(h_average);

    for(int i=0; i<MAX_LEVEL; i++){
	for(int k=0;k<channels;k++)
	    {
		free(left[i][k]);
		free(right[i][k]);
		
	    }
	free(left[i]);
	free(right[i]);
	
    }
    free(left);
    free(right);

    cudaDeviceReset();
    
    return disparity;
}

/* returns disparity image (x and y) and confidence map of the fovea + the left fovea (rgb) */
float ***MatchGPULib::matchStackPyramid(cv_bridge::CvImagePtr cv_ptrL, cv_bridge::CvImagePtr cv_ptrR, float ***leftFov, float ***rightFov){

    int height ,width ,step ,channels;
    uchar *data, *data2;
    float *h_Kernel, *h_average;
    float **image, **image2,**finDisp;
    float ***left, ***right;
    float ***disparity;
    float ***leftFoveated;
    float ***rightFoveated;

    h_Kernel    = (float *)malloc(KERNEL_LENGTH * sizeof(float));
    h_average    = (float *)malloc(KERNEL_LENGTH * sizeof(float));
    height = (int) cv_ptrL->image.rows; /* Tian uses rows as width */
    width = (int) cv_ptrL->image.cols; /**/

    channels = 3;
    step=cv_ptrL->image.step;
   
	data = (uchar *) malloc(height * width * channels *sizeof(uchar));
	data = (uchar *) cv_ptrL->image.data;
	data2 = (uchar *) malloc(height * width * channels *sizeof(uchar));
	data2 = (uchar *) cv_ptrR->image.data;
	
	image = (float**)malloc(channels * sizeof(float*));
	image2 = (float**)malloc(channels * sizeof(float*));
    for(int i=0; i<channels; i++){
		image[i] = (float*)malloc(height * width * sizeof(float));
		image2[i] = (float*)malloc(height * width * sizeof(float));
    }
    
    for(int k=0;k<channels;k++)
		for(int i=0;i<height;i++)
			for(int j=0;j<width;j++)
			{
				*(image[k]+i*width+j) = (float)*(data+i*step+j*channels+k);
				*(image2[k]+i*width+j) = (float)*(data2+i*step+j*channels+k);
			}
			
	//  1D gaussian kernel function
    this->gaussiankernel(h_Kernel);
	
    setConvolutionKernel(h_Kernel);
	*h_average=0.0;
	*(h_average+1)=0.3333;
	*(h_average+2)=0.3333;
	*(h_average+3)=0.3333;
	*(h_average+4)=0.0;

    setConvolutionAverageKernel(h_average);

	//Building a pyramid 
    left = CreatePyramidFromImage(image, channels, height, width, h_Kernel);
    right = CreatePyramidFromImage(image2, channels, height, width, h_Kernel);
    
	leftFoveated = CreateFoveatedPyramid(left, channels, height, width);
	rightFoveated = CreateFoveatedPyramid(right, channels, height, width);

/* DEBUG
    int heightp[MAX_LEVEL-1], widthp[MAX_LEVEL-1];
    int rows,cols;
    IplImage *outImg;char string[20];

    widthp[0] = width;
    heightp[0] = height;
    for(int i=0; i<MAX_LEVEL; i++)
    {
		widthp[i+1]=widthp[i]/SCALE;
		heightp[i+1]=heightp[i]/SCALE;
    }
    rows=widthp[foveatelevel-1];
    cols=heightp[foveatelevel-1];

    for(int level=MAX_LEVEL-1;level>=foveatelevel-1;level--)
    {
		leftFoveated[level] = left[level];     

/* DEBUG - prints left image from pyramid
	outImg = cvCreateImage(cvSize(widthp[level],heightp[level]), 8, 3);  
	for(int k=0;k<channels;k++)
	    for(int l=0;l<outImg->height;l++)
		for(int m=0;m<outImg->width;m++)
		{
		    *(outImg->imageData+l*outImg->widthStep+m*channels+k) = (uchar)*(leftFoveated[level][k]+l*outImg->width+m);
	
	        }
	sprintf(string,"LPyrfoveated%d.bmp",level+1);
	cvSaveImage(string,outImg); 
 
}

    for(int level=MAX_LEVEL-1;level>=foveatelevel-1;level--)
    {
		rightFoveated[level] = right[level];     

/* DEBUG  - prints right image from pyramid
	outImg = cvCreateImage(cvSize(widthp[level],heightp[level]), 8, 3);  
	for(int k=0;k<channels;k++)
	    for(int l=0;l<outImg->height;l++)
		for(int m=0;m<outImg->width;m++)
		{
		    *(outImg->imageData+l*outImg->widthStep+m*channels+k) = (uchar)*(rightFoveated[level][k]+l*outImg->width+m);
	
	        }
	sprintf(string,"RPyrfoveated%d.bmp",level+1);
	cvSaveImage(string,outImg); 
}*/


/* copy the data to the pointer array , we want to publish this on ros */
for (int m = 0; m < MAX_LEVEL ; m++){ /* 14 is max level, but foveatelevel is 7 */
    leftFov[m] = leftFoveated[m]; 
    rightFov[m] = rightFoveated[m]; 
}

	//Foveated Stereo Matching Part
	disparity = matching(leftFoveated, rightFoveated, channels, height, width);
	//finDisp=hierarchicalDisparity(right[0], disparity, channels, width, height);

/* DEBUG - prints disparity image
    for(int level=0;level<foveatelevel;level++)
    {
	outImg = cvCreateImage(cvSize(getFoveaWidth(),getFoveaHeight()), 8, 3);  

	for(int k=0;k<channels;k++)
	    for(int l=0;l<outImg->height;l++)
		for(int m=0;m<outImg->width;m++)
		{
		    *(outImg->imageData+l*outImg->widthStep+m*channels+k) = (uchar)*(disparity[level][k]+l*outImg->width+m);
	 

	        }
	sprintf(string,"foveatedDisp%d.bmp",level+1);
	cvSaveImage(string,outImg); 
}
*/
    
    free(h_Kernel);
    free(h_average);

    for (int j=0; j<channels; j++) {
        free(image[j]);
        free(image2[j]);
    }
    free(image);
    free(image2);

/* Don't
   for(int i=0; i<MAX_LEVEL; i++){
	for(int k=0;k<channels;k++)
	    {
		free(left[i][k]);
		free(right[i][k]);
		
	    }
	free(left[i]);
	free(right[i]);
	
    }
    free(left);
    free(right);
 */
    cudaDeviceReset();
  
// cvReleaseImage(&outImg); // valgrind (only if we used cvCreateImage)
    return disparity;
}
////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
/*int main(int argc, char **argv)
{

    IplImage* Img, *outImg;
    Img=cvLoadImage(argv[1],CV_LOAD_IMAGE_ANYCOLOR);

    int height ,width ,step ,channels;
    height    = Img->height;
    width     = Img->width;
    channels  = Img->nChannels;


    double tempTime = 0.0;
    char string1[20], string2[20];
    FILE * fp;


    data    = (uchar *)malloc(height * width * channels * sizeof(uchar));
    data  = (uchar*)Img->imageData;
}
*/


float MatchGPULib::gaussian(float x)
{
    float square,result;
    square = pow(SIGMA,2);
    result= exp(-(pow(x,2))/(2*square))/(sqrt(2*PI)*SIGMA);
    return result;
}

void MatchGPULib::gaussiankernel(float* k)
{
    float  weight[precision];
    float kern,tmp;

    int  mid = (int)(KERNEL_LENGTH / 2) +1;
    kern = 0.0;
    for(int i=0; i<KERNEL_LENGTH; i++)
    {
        *(k+i)=0;
        for(int n=0; n<precision; n++)
        {
	    weight[n]=0;
	    tmp=i+0.5-mid+(float)(n/((float)precision-1));
	    weight[n]=gaussian(tmp);
	    *(k+i)=*(k+i)+weight[n];
	}
        *(k+i)=*(k+i)/precision;
        kern=kern+*(k+i);
    }
    for(int i=0; i<KERNEL_LENGTH; i++)
    {
        *(k+i)=*(k+i)/kern;
//        printf("kernel[%d]=%f\n",i,*(k+i));
    }

*k=0.0816475;
*(k+1)=0.218507;
*(k+2)=0.303281;
*(k+3)=0.218507;
*(k+4)=0.0816475;
kern=0;
    for(int i=0; i<KERNEL_LENGTH; i++)
    {
        kern=kern+*(k+i);}
    for(int i=0; i<KERNEL_LENGTH; i++){
        *(k+i)=*(k+i)/kern;
        printf("kernel[%d]=%f\n",i,*(k+i));
    }
}

float ** MatchGPULib::convolutionCPU(float** im, float* h_Kernel, int channels, int imageW, int imageH)
{
    float
    *h_Input,
//    *h_OutputCPU,
    *h_Buffer;

    float** outim;

    outim = (float**)malloc(channels * sizeof(float*));
    for(int i=0; i<channels; i++){
	outim[i] = (float*)malloc(imageW * imageH * sizeof(float));
    }
    h_Input = (float*)malloc(imageW * imageH * sizeof(float));
//    h_OutputCPU = (float*)malloc(imageW * imageH * sizeof(float));
    h_Buffer = (float*)malloc(imageW * imageH * sizeof(float));

    for(int j=0;j<channels;j++) 
    {
	h_Input = im[j];

//	printf(" ...running convolutionRowCPU()\n");
	convolutionRowCPU(
            h_Buffer,
            h_Input,
            h_Kernel,
            imageW,
            imageH,
            KERNEL_RADIUS
	);

//	printf(" ...running convolutionColumnCPU()\n");
	convolutionColumnCPU(
            outim[j], //h_OutputCPU,
            h_Buffer,
            h_Kernel,
            imageW,
            imageH,
            KERNEL_RADIUS
	);

//	for(int i=0; i<imageW * imageH; i++)
//	    outim[j][i] = h_OutputCPU[i];
    }

    free(h_Input);
    free(h_Buffer);
    return outim;
}


float ** MatchGPULib::subsampledimageCPU(float**im, float scalefactor, int channels, int imageW, int imageH, int imageW2, int imageH2)
{
    float** outim, **temp;

    outim = (float**)malloc(channels * sizeof(float*));
    for(int i=0; i<channels; i++){
	outim[i] = (float*)malloc(imageW2 * imageH2 * sizeof(float));
    }
    temp = (float**)malloc(channels * sizeof(float*));
    for(int i=0; i<channels; i++){
	temp[i] = (float*)malloc(imageW2 * imageH * sizeof(float));
    }

   for(int k=0; k<channels; k++){
    for(int i=0; i<imageH; i++)
	for(int j=0; j<imageW2; j++)
	{
	    temp[k][i*imageW2+j]=im[k][i*imageW+(int)(j*scalefactor)]*(scalefactor*j-(int)(j*scalefactor))
				+im[k][i*imageW+(int)(j*scalefactor)+1]*(1-(scalefactor*j-(int)(j*scalefactor)));
	}

    for(int i=0; i<imageH2; i++)
	for(int j=0; j<imageW2; j++)
	{
	    outim[k][i*imageW2+j]=temp[k][(int)(i*scalefactor)*imageW2+j]*(i*scalefactor-(int)(i*scalefactor))
				+temp[k][(int)(i*scalefactor+1)*imageW2+j]*(1-(i*scalefactor-(int)(i*scalefactor)));
	}
   }

    for (int j=0; j<channels; j++) {
        free(temp[j]);
    }
    free(temp);

    return outim;
}


float ** MatchGPULib::convolutionGPU(float** im, int channels, int imageW, int imageH)
{
    float
    *d_Input,
    *d_Output,
    *d_Buffer;

    float** outim;
    int  tempW, tempH;
    double gpuTime;

    StopWatchInterface *hTimer = NULL;
    sdkCreateTimer(&hTimer);
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
    outim = (float**)malloc(channels * sizeof(float*));
    for(int i=0; i<channels; i++){
	outim[i] = (float*)malloc(imageW * imageH * sizeof(float));
    } 

    tempW=iDivUp(imageW, COLUMNS_BLOCKDIM_X)*COLUMNS_BLOCKDIM_X;
    tempH=iDivUp(imageH, (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y))*(COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y);

    checkCudaErrors(cudaMalloc((void **)&d_Input,   imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_Output,  imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_Buffer , tempW * tempH * sizeof(float)));

	sdkStopTimer(&hTimer);
	gpuTime = 0.001 * sdkGetTimerValue(&hTimer);
//printf("gpuTime = %.5f \n",gpuTime);

    for(int j=0;j<channels;j++) 
    {
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	checkCudaErrors(cudaMemcpy(d_Input, im[j], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&hTimer);
	gpuTime = 0.001 * sdkGetTimerValue(&hTimer);
//printf("gpuTime = %.5f \n",gpuTime);
	excutionTime[8] = gpuTime + excutionTime[8];

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	convolutionRowsGPU(
	    d_Buffer,
	    d_Input,
	    imageW,
	    imageH
        );
	checkCudaErrors(cudaDeviceSynchronize());

        convolutionColumnsGPU(
	    d_Output,
	    d_Buffer,
	    imageW,
	    imageH
        );

	checkCudaErrors(cudaDeviceSynchronize());

	sdkStopTimer(&hTimer);
	gpuTime = 0.001 * sdkGetTimerValue(&hTimer);
	excutionTime[7] = gpuTime + excutionTime[7];

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	checkCudaErrors(cudaMemcpy(outim[j], d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));
	sdkStopTimer(&hTimer);
	gpuTime = 0.001 * sdkGetTimerValue(&hTimer);
	excutionTime[8] = gpuTime + excutionTime[8];
    }

//	printf("convolutionSeparable, Throughput = %.4f MPixels/sec, Time = %.5f s, Size = %u Pixels, NumDevsUsed = %i, Workgroup = %u\n",
//	    (1.0e-6 * (double)(imageW * imageH * channels)/ sumTime), sumTime, (imageW * imageH * channels), 1, 0);

//	printf("\nReading back GPU results...\n\n");

	sdkResetTimer(&hTimer);
//	sdkStartTimer(&hTimer);

    checkCudaErrors(cudaFree(d_Buffer));
    checkCudaErrors(cudaFree(d_Output));
    checkCudaErrors(cudaFree(d_Input));

//	sdkStopTimer(&hTimer);
//	gpuTime = 0.001 * sdkGetTimerValue(&hTimer);
//printf("gpuTime = %.5f \n",gpuTime);

sdkDeleteTimer(&hTimer); // valgrind
    return outim;
	
}



float ** MatchGPULib::subsampledimageGPU(float**im, float scalefactor, int channels, int imageW, int imageH, int imageW2, int imageH2)
{
    float** outim;
    float *d_Output;
    double gpuTime;

    StopWatchInterface *hTimer = NULL;
    sdkCreateTimer(&hTimer);

    cudaArray
    *a_Src;

    cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();

    outim = (float**)malloc(channels * sizeof(float*));
    for(int i=0; i<channels; i++){
	outim[i] = (float*)malloc(imageW2 * imageH2 * sizeof(float));
    }


//    checkCudaErrors(cudaMalloc((void **)&d_Input,   imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_Output,  imageW2 * imageH2 * sizeof(float)));
    checkCudaErrors(cudaMallocArray(&a_Src, &floatTex, imageW, imageH));


   for(int j=0; j<channels; j++){

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
//	checkCudaErrors(cudaMemcpy(d_Input, im[j], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, im[j], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
	sdkStopTimer(&hTimer);
	gpuTime = 0.001 * sdkGetTimerValue(&hTimer);
	excutionTime[9] = gpuTime + excutionTime[9];

	checkCudaErrors(cudaDeviceSynchronize());

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	subsampleGPU(
	    d_Output,
	    a_Src,
	    imageW,
	    imageH,
	    scalefactor,
	    imageW2,
	    imageH2
        );
	sdkStopTimer(&hTimer);
	gpuTime = 0.001 * sdkGetTimerValue(&hTimer);
	excutionTime[10] = gpuTime + excutionTime[10];

	checkCudaErrors(cudaDeviceSynchronize());

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	checkCudaErrors(cudaMemcpy(outim[j], d_Output, imageW2 * imageH2 * sizeof(float), cudaMemcpyDeviceToHost));
	sdkStopTimer(&hTimer);
	gpuTime = 0.001 * sdkGetTimerValue(&hTimer);
	excutionTime[9] = gpuTime + excutionTime[9];
    }

    cudaFreeArray(a_Src);
    cudaFree(d_Output);

sdkDeleteTimer(&hTimer); // valgrind
    return outim;
}

float *** MatchGPULib::CreatePyramidFromImage(float **im, int channels, int heightInit, int widthInit, float *h_Kernel)
{
    StopWatchInterface *hTimer = NULL;
    sdkCreateTimer(&hTimer);
    int height, width, height2, width2;
    float scalefactors;
//    float **pyramid[MAX_LEVEL][LAYER];
    float ****pyramid;
//    float **DOGpyramid[MAX_LEVEL][LAYER];
float ***p;

    height=heightInit;
    width=widthInit;


    pyramid = (float****)malloc(MAX_LEVEL * sizeof(float***));
    p = (float***)malloc(MAX_LEVEL * sizeof(float**));
    for(int i=0; i<MAX_LEVEL; i++)
    {
	pyramid[i] = (float***)malloc(LAYER * sizeof(float**));

    }


    height    = heightInit;
    width     = widthInit;


    pyramid[0][0]=im;

    for(int i=0; i<MAX_LEVEL; i++)
    {
	for(int j=0;j<LAYER-1;j++)
	{

	    sdkResetTimer(&hTimer);
	    sdkStartTimer(&hTimer);

	    pyramid[i][j+1]=convolutionGPU(pyramid[i][j],channels,width,height);

	    
	    sdkStopTimer(&hTimer);
	    excutionTime[3] = 0.001 * sdkGetTimerValue(&hTimer) + excutionTime[3];

	}

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	if(i==0){
	    scalefactors=SCALE; width2=width/SCALE; height2=height/SCALE;
	    pyramid[i+1][0]=subsampledimageGPU(pyramid[i][LAYER-1],scalefactors,channels,width,height,width2,height2);
	}
	if(i<MAX_LEVEL-2){
//	    scalefactors=SCALE*SCALE;
	    scalefactors=0.000+(int)(SCALE*SCALE+0.5);
	    width2=width/SCALE; height2=height/SCALE;
	    width2=width2/SCALE; height2=height2/SCALE;
	    pyramid[i+2][0]=subsampledimageGPU(pyramid[i][LAYER-1],scalefactors,channels,width,height,width2,height2);
	}


	width=width/SCALE;
	height=height/SCALE;

	sdkStopTimer(&hTimer);
	excutionTime[4] = 0.001 * sdkGetTimerValue(&hTimer) + excutionTime[4];


    }

    if(useGPU == 1){
	printf("Running time for convolution only on GPU is %.5f s.\n\n",excutionTime[7]);
	printf("Running time for convolution part Memcopy between CPU and GPU is %.5f s.\n\n",excutionTime[8]);
    }
    printf("Totally %d times convolution time is %.5f s.\n\n",((MAX_LEVEL-1)*(LAYER-1)),excutionTime[3]);
    if(useGPU == 1){
	printf("Running time for subsample only on GPU is %.5f s.\n\n",excutionTime[10]);
	printf("Running time for subsample part Memcopy between CPU and GPU is %.5f s.\n\n",excutionTime[9]);
    }
    printf("Totally %d times subsample time is %.5f s.\n\n",(MAX_LEVEL-1),excutionTime[4]);

    for(int i=0; i<MAX_LEVEL; i++){
	p[i]=pyramid[i][0];
    }

    sdkDeleteTimer(&hTimer); // valgrind
    return p;
}


float *** MatchGPULib::CreateFoveatedPyramid(float ***im, int channels, int heightInit, int widthInit)
{
    int rows, cols, level, x,y,x1,y1,l,r,u,d;
    float ***foveated;
    int height[MAX_LEVEL-1], width[MAX_LEVEL-1];
    char string[20];
    IplImage *outImg;

    width[0] = widthInit;
    height[0] = heightInit;
    for(int i=0; i<MAX_LEVEL; i++)
    {
		width[i+1]=width[i]/SCALE;
		height[i+1]=height[i]/SCALE;
    }
    rows=width[foveatelevel-1];
    cols=height[foveatelevel-1];
    x1=rows/2;
    y1=cols/2;

    foveated = (float***)malloc(MAX_LEVEL * sizeof(float**));

    for(level=MAX_LEVEL-1;level>=foveatelevel-1;level--)
    {
		foveated[level] = im[level];     

/* DEBUG 
	outImg = cvCreateImage(cvSize(width[level],height[level]), 8, 3);  
	for(int k=0;k<channels;k++)
	    for(int l=0;l<outImg->height;l++)
		for(int m=0;m<outImg->width;m++)
		{
		    *(outImg->imageData+l*outImg->widthStep+m*channels+k) = (uchar)*(foveated[level][k]+l*outImg->width+m);
	 //         printf("%f\n", *(left[i][k]+l*Img->width+m));
	        }
	sprintf(string,"Pyrfoveated%d.bmp",level+1);
	cvSaveImage(string,outImg); 
*/
      
    } 

//cvReleaseImage(&outImg); // valgrind

    for(level=foveatelevel-2;level>=0;level--)
    {
		x=width[level];
		y=height[level];
		l=x/2-x1;
		u=y/2-y1;
     //printf("level=%d : x= %d, y = %d, l = %d , u = %d, x1 = %d, y1 = %d,rows = %d, cols = %d\n",level,x,y,l,u,x1,y1,rows,cols); // debugging purpose
		foveated[level] = (float**)malloc(channels * sizeof(float*));
		for(int k=0;k<channels;k++){
			foveated[level][k] = (float*)malloc(rows * cols * sizeof(float));
			for(int i=0;i<cols;i++){
				memcpy(&foveated[level][k][i*rows], &im[level][k][(u+i)*x+l], rows * sizeof(float));
			}
		}
    }


    return foveated;

}





float *** MatchGPULib::matching(float ***iml, float ***imr, int channels, int heightInit, int widthInit)
{
/***/
    uint64_t diff,dif;
    struct timespec start, end;
/***/

        float ***disp, ***dispOut;;
    int height[MAX_LEVEL-1], width[MAX_LEVEL-1];
   // float **warpRight, **temp;
    int i,k,mi;
    float scalefactor, step;
    float *oldDH, *oldDV;
    int height2, width2;

    width[0] = widthInit;
    height[0] = heightInit;
    scalefactor=1/SCALE;

    for(i=0; i<MAX_LEVEL; i++)
    {
        width[i+1]=width[i]/SCALE;
        height[i+1]=height[i]/SCALE;
    }
    if(foveatedmatching==1){
        width2=width[foveatelevel-2];
        height2=height[foveatelevel-2];
        fovH=height[foveatelevel-1];
	    fovW=width[foveatelevel-1];
        for(int i=0; i<foveatelevel-1; i++)
        {
            width[i]=width[foveatelevel-1];
            height[i]=height[foveatelevel-1];
        }
    }
    disp = (float***)malloc(MAX_LEVEL * sizeof(float**));
    //    dispOut = (float***)malloc(MAX_LEVEL * sizeof(float**));
    for(i=0; i<MAX_LEVEL; i++)
    {
        disp[i] = (float**)malloc(3 * sizeof(float*));
        //	dispOut[i] = (float**)malloc(channels * sizeof(float*));
        for(k=0;k<3;k++){
//            disp[i][k] = (float*)malloc(height[i] * width[i] * sizeof(float));
	checkCudaErrors( cudaMallocHost((void**)&disp[i][k], height[i] * width[i] * sizeof(float)) ); 
            //	    dispOut[i][k] = (float*)malloc(height[i] * width[i] * sizeof(float));
        }
    }

    for(i=MAX_LEVEL-1; i>=0; i--)
    {
//        oldDH = (float*)malloc(height[i] * width[i] * sizeof(float));
//        oldDV = (float*)malloc(height[i] * width[i] * sizeof(float));
	checkCudaErrors( cudaMallocHost((void**)&oldDH, height[i] * width[i] * sizeof(float)) ); 
	checkCudaErrors( cudaMallocHost((void**)&oldDV, height[i] * width[i] * sizeof(float)) ); 

        printf("level[%d] \n",i+1);
        int mi=(i>5)?levelcutoff:((i+1)*2);//(i+1);
        step=1.0;
//        for(int j=1;j<=mi;j++){

        for(int j=1;j<=1;j++){
clock_gettime(CLOCK_MONOTONIC, &start); /* mark start time */
            disp[i] = matchlevel(iml[i],imr[i],disp[i],channels,width[i],height[i],MAX_LEVEL-1-i, j, step);
clock_gettime(CLOCK_MONOTONIC, &end); /* mark the end time */
dif = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
cout << "Matching level "<< i << " took "<< dif/1.0e6 << " ms"<< endl;
/*            if(j%2==1){
//                memcpy(oldDH,disp[i][0],height[i] * width[i] * sizeof(float));
//                memcpy(oldDV,disp[i][1],height[i] * width[i] * sizeof(float));
            }
            else{
                if((mi/2-j/2)<7){ step=((mi/2-j/2)-1)*((1-0.1)/(mi/2-1.0))+0.1; }
                else{step=1.0;}//printf("step=%f\n",step);

///////////////////MATLAB CODE///////////////////////
/*
    step_initial=1.0;
    step_finish=0.1;
    cut_iteration=7;
    
    if(numIterations==1)
        ss=step_initial;
        return;
    elseif ((numIterations-currentIteration) < cut_iteration)
        ss=(numIterations-currentIteration-1)*((step_initial-step_finish)/(numIterations-1.0)) + step_finish;
        return;
    end
    ss=step_initial;
*/
/////////////////MATLAB CODE END//////////////////////

 /*               if (differenceIterations(disp[i][0],disp[i][1],disp[i][2],oldDH,oldDV,0.1*0.1,width[i],height[i])==0){
                    printf("STOPPED level: %i after %i cycles !!!\n",i,j);
                    break;
                }
*/
//            }

//            disp[i] = convolutionGPUTa(disp[i], 3, width[i],height[i]);

        }
	
////cudaMemGetInfo(&freeMem, &totalMem);
//printf("Free Memory:%luMB = %luKB,\tTotal Memory:%luMB = %luKB\n",freeMem/1024/1024,freeMem/1024,totalMem/1024/1024,totalMem/1024);

        if(i>0){
            if(foveatedmatching==0){

 clock_gettime(CLOCK_MONOTONIC, &start); /* mark start time */
                disp[i-1]=subsampleDisp(disp[i], scalefactor, 3, width[i], height[i], width[i-1], height[i-1]);
 clock_gettime(CLOCK_MONOTONIC, &end); /* mark the end time */
            }
            else{
                if(i>=foveatelevel){
 clock_gettime(CLOCK_MONOTONIC, &start); /* mark start time */
               disp[i-1]=subsampleDisp(disp[i], scalefactor, 3, width[i], height[i], width[i-1], height[i-1]);
clock_gettime(CLOCK_MONOTONIC, &end); /* mark the end time */
                 }
                else{
 clock_gettime(CLOCK_MONOTONIC, &start); /* mark start time */
               disp[i-1]=foveatedsubsampleDisp(disp[i], scalefactor, 3, width[i], height[i], width2, height2);
clock_gettime(CLOCK_MONOTONIC, &end); /* mark the end time */

                  }
            }
diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
cout << "Subsampling level "<< i << " took "<< diff/1.0e6 << " ms"<< endl;
cout << "Matching level "<< i << " in total took "<< (diff+dif)/1.0e6 << " ms"<< endl;
        }
////cudaMemGetInfo(&freeMem, &totalMem);
//printf("Free Memory:%luMB = %luKB,\tTotal Memory:%luMB = %luKB\n",freeMem/1024/1024,freeMem/1024,totalMem/1024/1024,totalMem/1024);
        //	dispOut[i] = warpRightImage(imr[i],disp[i],channels,width[i],height[i]);
//        printf("level[%d] second time\n",i+1);
////cudaMemGetInfo(&freeMem, &totalMem);
//printf("Free Memory:%luMB = %luKB,\tTotal Memory:%luMB = %luKB\n",freeMem/1024/1024,freeMem/1024,totalMem/1024/1024,totalMem/1024);

//        free(oldDH);
//        free(oldDV);
    cudaFreeHost(oldDH);
    cudaFreeHost(oldDV);
    }

    return disp;
}




int MatchGPULib::differenceIterations(float*DH,float*DV,float*conf,float*OldDH,float*OldDV,float threshold, int imageW, int imageH)
{
    int isDif = 1;
    float Dif1, Dif2;
    Dif1 = weightedDifference(DH,OldDH,conf,imageW,imageH);
    Dif2 = weightedDifference(DV,OldDV,conf,imageW,imageH);
    if ((Dif1<threshold) && (Dif2<threshold)){
        isDif = 0;}

    return isDif;
}

float MatchGPULib::weightedDifference(float*D,float*OldD,float*conf, int imageW, int imageH)
{
    float *d_temp, *d_odata, *weightedDif, *d_D, *d_OldD, *d_conf;
    float sumPix=0;
    float theDif=0;
    int numBlocks, numBlocks2;

    int threads = 256;
    numBlocks = (imageW * imageH + (threads * 2 - 1)) / (threads * 2);

    weightedDif = (float*)malloc(numBlocks * sizeof(float));
    checkCudaErrors(cudaMalloc((void **)&d_temp,  imageW * imageH * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **)&d_D,   imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_OldD,  imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_conf , imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_D, D, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_OldD, OldD, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conf, conf, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));


    checkCudaErrors(cudaMalloc((void **)&d_odata,  numBlocks * sizeof(float)));

    checkCudaErrors(cudaDeviceSynchronize());

    weightedDifferenceGPU(
	    d_temp,
	    d_D,
	    d_OldD,
	    d_conf,
	    imageW,
	    imageH
        );


    reduceGPU(
	    d_odata,
	    d_temp,
	    numBlocks,
	    threads,
	    imageW * imageH
        );

    checkCudaErrors(cudaMemcpy(d_temp, d_odata, numBlocks * sizeof(float), cudaMemcpyDeviceToDevice));

    reduceGPU(
	    d_odata,
	    d_conf,
	    numBlocks,
	    threads,
	    imageW * imageH
        );

    checkCudaErrors(cudaMemcpy(d_conf, d_odata, numBlocks * sizeof(float), cudaMemcpyDeviceToDevice));

    numBlocks2 = numBlocks;
    numBlocks = (numBlocks2 + (threads * 2 - 1)) / (threads * 2);

    reduceGPU(
	    d_odata,
	    d_temp,
	    numBlocks,
	    threads,
	    numBlocks2
        );
    checkCudaErrors(cudaMemcpy(weightedDif, d_odata, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));

    reduceGPU(
	    d_odata,
	    d_conf,
	    numBlocks,
	    threads,
	    numBlocks2
        );
    checkCudaErrors(cudaMemcpy(conf, d_odata, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));

//printf("numBlocks2 = %d\n",numBlocks2);
//printf("numBlocks = %d\n",numBlocks);

    for(int i=0;i<numBlocks;i++){
	sumPix = sumPix + weightedDif[i];
	theDif = theDif + conf[i];
    }


/*    for(int i=0;i<imageH*imageW;i++){
	sumPix = sumPix + weightedDif[i];
	theDif = theDif + conf[i];
    }*/

    theDif = sumPix/theDif;
//printf("The Difference = %f\n",theDif);

    cudaFree(d_D);
    cudaFree(d_OldD);
    cudaFree(d_conf);
    cudaFree(d_temp);
    cudaFree(d_odata);
    free(weightedDif);

    return theDif;
}







float ** MatchGPULib::warpRightImage(float **right, float **disparity, int channels, int imageW, int imageH)
{
    float** outim;
    float *d_Output, *temp;
    double gpuTime;

    StopWatchInterface *hTimer = NULL;
    sdkCreateTimer(&hTimer);

    cudaArray
    *a_Src, *dispx, *dispy;

    cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();

    outim = (float**)malloc(channels * sizeof(float*));
    for(int i=0; i<channels; i++){
	outim[i] = (float*)malloc(imageW * imageH * sizeof(float));
    }

    checkCudaErrors(cudaMalloc((void **)&d_Output,  imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&temp,  imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMallocArray(&a_Src, &floatTex, imageW, imageH));
    checkCudaErrors(cudaMallocArray(&dispx, &floatTex, imageW, imageH));
    checkCudaErrors(cudaMallocArray(&dispy, &floatTex, imageW, imageH));
    checkCudaErrors(cudaMemcpyToArray(dispx, 0, 0, disparity[0], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToArray(dispy, 0, 0, disparity[1], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));


   for(int j=0; j<channels; j++){

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, right[j], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
	sdkStopTimer(&hTimer);
	gpuTime = 0.001 * sdkGetTimerValue(&hTimer);
	excutionTime[11] = gpuTime + excutionTime[11];

	checkCudaErrors(cudaDeviceSynchronize());

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

	warp(
	    d_Output,
	    a_Src,
	    dispx,
	    dispy,
	    imageW,
	    imageH
        );

	sdkStopTimer(&hTimer);
	gpuTime = 0.001 * sdkGetTimerValue(&hTimer);
	excutionTime[12] = gpuTime + excutionTime[12];

	checkCudaErrors(cudaDeviceSynchronize());

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	checkCudaErrors(cudaMemcpy(outim[j], d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));
	sdkStopTimer(&hTimer);
	gpuTime = 0.001 * sdkGetTimerValue(&hTimer);
	excutionTime[11] = gpuTime + excutionTime[11];
    }

    cudaFreeArray(a_Src);
    cudaFreeArray(dispx);
    cudaFreeArray(dispy);
    cudaFree(temp);
    cudaFree(d_Output);

sdkDeleteTimer(&hTimer); // valgrind
    return outim;
}







float ** MatchGPULib::subsampleDisp(float**im, float scalefactor, int channels, int imageW, int imageH, int imageW2, int imageH2)
{
    float** outim;
    float *d_Output;
    double gpuTime;

    StopWatchInterface *hTimer = NULL;
    sdkCreateTimer(&hTimer);

    cudaArray
    *a_Src;

    cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();

    outim = (float**)malloc(channels * sizeof(float*));
    for(int i=0; i<channels; i++){
	outim[i] = (float*)malloc(imageW2 * imageH2 * sizeof(float));
    }

    checkCudaErrors(cudaMalloc((void **)&d_Output,  imageW2 * imageH2 * sizeof(float)));
    checkCudaErrors(cudaMallocArray(&a_Src, &floatTex, imageW, imageH));


   for(int j=0; j<channels; j++){

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, im[j], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
	sdkStopTimer(&hTimer);
	gpuTime = 0.001 * sdkGetTimerValue(&hTimer);
	excutionTime[9] = gpuTime + excutionTime[9];

	checkCudaErrors(cudaDeviceSynchronize());

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	subsampleDispGPU(
	    d_Output,
	    a_Src,
	    imageW,
	    imageH,
	    scalefactor,
	    imageW2,
	    imageH2
        );
	sdkStopTimer(&hTimer);
	gpuTime = 0.001 * sdkGetTimerValue(&hTimer);
	excutionTime[10] = gpuTime + excutionTime[10];

	checkCudaErrors(cudaDeviceSynchronize());

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	checkCudaErrors(cudaMemcpy(outim[j], d_Output, imageW2 * imageH2 * sizeof(float), cudaMemcpyDeviceToHost));
	sdkStopTimer(&hTimer);
	gpuTime = 0.001 * sdkGetTimerValue(&hTimer);
	excutionTime[9] = gpuTime + excutionTime[9];
    }

    cudaFreeArray(a_Src);
    cudaFree(d_Output);

sdkDeleteTimer(&hTimer); // valgrind
    return outim;
}




float ** MatchGPULib::foveatedsubsampleDisp(float**im, float scalefactor, int channels, int imageW, int imageH, int imageW2, int imageH2)
{
    float** outim, **temp;
    float *d_Output;
    int x1,y1,l,u;

    cudaArray
    *a_Src;

    cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();

    outim = (float**)malloc(channels * sizeof(float*));
    temp = (float**)malloc(channels * sizeof(float*));
    for(int i=0; i<channels; i++){
	outim[i] = (float*)malloc(imageW * imageH * sizeof(float));
	temp[i] = (float*)malloc(imageW2 * imageH2 * sizeof(float));
    }
    x1=imageW/2;
    y1=imageH/2;
	l=imageW2/2-x1;
	u=imageH2/2-y1;

    checkCudaErrors(cudaMalloc((void **)&d_Output,  imageW2 * imageH2 * sizeof(float)));
    checkCudaErrors(cudaMallocArray(&a_Src, &floatTex, imageW, imageH));


   for(int j=0; j<channels; j++){

	checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, im[j], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));


	checkCudaErrors(cudaDeviceSynchronize());

	subsampleDispGPU(
	    d_Output,
	    a_Src,
	    imageW,
	    imageH,
	    scalefactor,
	    imageW2,
	    imageH2
        );


	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(temp[j], d_Output, imageW2 * imageH2 * sizeof(float), cudaMemcpyDeviceToHost));
	for(int i=0;i<imageH;i++){
	    memcpy(&outim[j][i*imageW], &temp[j][(u+i)*imageW2+l], imageW * sizeof(float));
	}
    }

for (int j=0; j<channels; j++) {
    free(temp[j]);
}
free(temp);
    cudaFreeArray(a_Src);
    cudaFree(d_Output);

    return outim;
}






float ** MatchGPULib::matchlevel(float **left, float **right, float **disparity, int channels, int imageW, int imageH, int level, int iter, float threshold)
{

    float  *** direction;
    float *l, *r, *u, *d, *c, *h_Kernel;
    float *temp, *directionl, *directionr, *directionu, *directiond, *directionc, *disp0, *disp1, *disp2, *imleft2, *imright2;
    double gpuTime;
    int realSmoothtime;
    float thresholdx, thresholdy,thresholdtest;
//    Point move[5]={{0-thresholdx,0},{thresholdx,0},{0,0-thresholdy},{0,thresholdy},{0,0}};
    thresholdtest=threshold;
    threshold=1.0;
    thresholdx=threshold;
    thresholdy=threshold;

    Point move[5]={{0.0-thresholdx,0.0},{thresholdx,0.0},{0.0,0.0-thresholdy},{0.0,thresholdy},{0.0,0.0}};
    
    StopWatchInterface *hTimer = NULL;
    sdkCreateTimer(&hTimer);

    cudaArray
    *a_Src, *compare, *texturel, *texturer, *texturelr, *temp2, *imleft, *imright;

    cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();

    h_Kernel = (float*)malloc(5 * sizeof(float));


    checkCudaErrors(cudaMalloc((void **)&l,  imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&r,  imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&u,  imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d,  imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&c,  imageW * imageH * sizeof(float)));

    checkCudaErrors(cudaMallocArray(&a_Src, &floatTex, imageW, imageH));
    checkCudaErrors(cudaMallocArray(&compare, &floatTex, imageW, imageH));
    checkCudaErrors(cudaMallocArray(&texturel, &floatTex, imageW, imageH));
    checkCudaErrors(cudaMallocArray(&texturer, &floatTex, imageW, imageH));
    checkCudaErrors(cudaMallocArray(&texturelr, &floatTex, imageW, imageH));

if(usingMoreGPUMemory==1){
    checkCudaErrors(cudaMalloc((void **)&directionl,  imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&directionr,  imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&directionu,  imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&directiond,  imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&directionc,  imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMallocArray(&temp2, &floatTex, imageW, imageH));

    checkCudaErrors(cudaMalloc((void **)&disp0,  imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&disp1,  imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&disp2,  imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&imleft2,  imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&imright2,  imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMallocArray(&imleft, &floatTex, imageW, imageH));
    checkCudaErrors(cudaMallocArray(&imright, &floatTex, imageW, imageH));
}
else{
    direction = (float***)malloc(5 * sizeof(float**));
    for(int i=0; i<5; i++){
        direction[i] = (float**)malloc(channels * sizeof(float*));
        for(int j=0; j<channels; j++){
//            direction[i][j] = (float*)malloc(imageW * imageH * sizeof(float));
	checkCudaErrors( cudaMallocHost((void**)&direction[i][j], imageW * imageH * sizeof(float)) ); 
        }
    }
}

//    checkCudaErrors(cudaMallocArray(&dispy, &floatTex, imageW, imageH));
////cudaMemGetInfo(&freeMem, &totalMem);
//printf("Free Memory:%luMB = %luKB,\tTotal Memory:%luMB = %luKB\n",freeMem/1024/1024,freeMem/1024,totalMem/1024/1024,totalMem/1024);

/*
*h_Kernel=0.090354;
*(h_Kernel+1)=0.241821;
*(h_Kernel+2)=0.335650;
*(h_Kernel+3)=0.241821;
*(h_Kernel+4)=0.090354;
*/
//int mi=(level<15)?1:1;//(MAX_LEVEL-level);
int mi=((13-level)>5)?levelcutoff:((13-level+1)*2);
//printf("level=%d\tmi=%d\n",level,mi);
for(int m=1; m<=mi; m ++){

   for(int j=0; j<channels; j++){


if(usingMoreGPUMemory==1){
    
    sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
    
	checkCudaErrors(cudaMemcpyToArray(compare, 0, 0, right[j], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, left[j], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));

    sdkStopTimer(&hTimer);
    gpuTime = 0.001 * sdkGetTimerValue(&hTimer);
    excutionTime[13] = gpuTime + excutionTime[13];
    
	if(m==1){
//	checkCudaErrors(cudaMemcpyToArray(imright, 0, 0, right[j], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
//	checkCudaErrors(cudaMemcpyToArray(imleft, 0, 0, left[j], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(disp0, disparity[0], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(disp1, disparity[1], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
//	checkCudaErrors(cudaMemcpy(imright2, right[j], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
//	checkCudaErrors(cudaMemcpy(imleft2, left[j], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaDeviceSynchronize());
	}
//	else{
    checkCudaErrors(cudaMemcpyToArray(texturel, 0, 0, disp0, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToArray(texturer, 0, 0, disp1, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
//compare=imright;
//a_Src=imleft;
//	checkCudaErrors(cudaMemcpyToArray(compare, 0, 0, imright2, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
//	checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, imleft2, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaDeviceSynchronize());
//	}

}
else{
    checkCudaErrors(cudaMemcpyToArray(texturel, 0, 0, disparity[0], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToArray(texturer, 0, 0, disparity[1], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToArray(compare, 0, 0, right[j], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, left[j], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));


	checkCudaErrors(cudaDeviceSynchronize());
}

       
	warp(
	    c,
	    compare,
	    texturel,
	    texturer,
	    imageW,
	    imageH
        );


	checkCudaErrors(cudaMemcpyToArray(compare, 0, 0, c, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));

	checkCudaErrors(cudaDeviceSynchronize());

//	sdkResetTimer(&hTimer);
//	sdkStartTimer(&hTimer);

	compareSquareIm(
	    u,
	    a_Src,
	    imageW,
	    imageH
        );

	compareSquareIm(
	    d,
	    compare,
	    imageW,
	    imageH
        );


	checkCudaErrors(cudaDeviceSynchronize());
/*
       convolutionRowsGPU(
                          l,
                          u,
                          imageW,
                          imageH
                          );
       checkCudaErrors(cudaDeviceSynchronize());
       
       convolutionColumnsGPU(
                             u,
                             l,
                             imageW,
                             imageH
                             );
       
       checkCudaErrors(cudaDeviceSynchronize());
       
       convolutionRowsGPU(
                          r,
                          d,
                          imageW,
                          imageH
                          );
       checkCudaErrors(cudaDeviceSynchronize());
       
       convolutionColumnsGPU(
                             d,
                             r,
                             imageW,
                             imageH
                             );
       
       checkCudaErrors(cudaDeviceSynchronize());
*/



    checkCudaErrors(cudaMemcpyToArray(texturel, 0, 0, u, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

        convolutionRowsGPUT(
            l,
            texturel,
            imageW,
            imageH
        );
    checkCudaErrors(cudaMemcpyToArray(texturelr, 0, 0, l, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

        convolutionColumnsGPUT(
            u,
            texturelr,
            imageW,
            imageH
        );


    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpyToArray(texturel, 0, 0, d, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

        convolutionRowsGPUT(
            r,
            texturel,
            imageW,
            imageH
        );
    checkCudaErrors(cudaMemcpyToArray(texturelr, 0, 0, r, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

        convolutionColumnsGPUT(
            d,
            texturelr,
            imageW,
            imageH
        );
    checkCudaErrors(cudaDeviceSynchronize());

       checkCudaErrors(cudaMemcpyToArray(texturel, 0, 0, u, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
       checkCudaErrors(cudaMemcpyToArray(texturer, 0, 0, d, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));

//////////////////////////////////////

       
       for (int i=0; i<5;i++){
           thresholdx=move[i].x;
           thresholdy=move[i].y;
//              printf("thresholdx=%f\tthresholdy=%f\n",move[i].x,move[i].y);
//            printf("thresholdx=%f\tthresholdy=%f\n",thresholdx,thresholdy);

           
       compareImMove(
                     c,
                     a_Src,
                     compare,
                     imageW,
                     imageH,
                     thresholdx,
                     thresholdy
                     );
       
       checkCudaErrors(cudaDeviceSynchronize());

           sdkResetTimer(&hTimer);
           sdkStartTimer(&hTimer);
           
           convolutionRowsGPU(
                              l,
                              c,
                              imageW,
                              imageH
                              );
           checkCudaErrors(cudaDeviceSynchronize());
           
           convolutionColumnsGPU(
                                 c,
                                 l,
                                 imageW,
                                 imageH
                                 );
           //printf("CudaErrors test 1\n");
           checkCudaErrors(cudaDeviceSynchronize());

       
/*       checkCudaErrors(cudaMemcpyToArray(texturelr, 0, 0, c, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
       checkCudaErrors(cudaDeviceSynchronize());
       
       convolutionRowsGPUT(
                           l,
                           texturelr,
                           imageW,
                           imageH
                           );
       checkCudaErrors(cudaDeviceSynchronize());
       checkCudaErrors(cudaMemcpyToArray(texturelr, 0, 0, l, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
       checkCudaErrors(cudaDeviceSynchronize());
       
       convolutionColumnsGPUT(
                              c,
                              texturelr,
                              imageW,
                              imageH
                              );
           
       checkCudaErrors(cudaDeviceSynchronize());
 */
       checkCudaErrors(cudaMemcpyToArray(texturelr, 0, 0, c, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
//       printf("threshold=%f\n",threshold);

           sdkStopTimer(&hTimer);
           gpuTime = 0.001 * sdkGetTimerValue(&hTimer);
           excutionTime[14] = gpuTime + excutionTime[14];
           
if(usingMoreGPUMemory==0){
	
       calculateImMoveCorr(
                           l,
                           texturel,
                           texturer,
                           texturelr,
                           imageW,
                           imageH,
                           thresholdx,
                           thresholdy
                           );
       
       checkCudaErrors(cudaDeviceSynchronize());
}
           
           

        
           
if(usingMoreGPUMemory==1){
switch( i ) 
{
    case 0 :
        temp = directionl;
        break;
    case 1 :
        temp = directionr;
        break;
    case 2 :
        temp = directionu;
        break;
    case 3 :
        temp = directiond;
        break;
    case 4 :
        temp = directionc;
        break;
}
    if(j!=0){
    calculateImMoveCorr(
                        l,
                        texturel,
                        texturer,
                        texturelr,
                        imageW,
                        imageH,
                        thresholdx,
                        thresholdy
                        );
    
    checkCudaErrors(cudaDeviceSynchronize());
    }
    
if(j==0){ //checkCudaErrors(cudaMemcpy(temp, l, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
    calculateImMoveCorr(
                        temp,
                        texturel,
                        texturer,
                        texturelr,
                        imageW,
                        imageH,
                        thresholdx,
                        thresholdy
                        );
    
    checkCudaErrors(cudaDeviceSynchronize());

}
else if(j==1){
	checkCudaErrors(cudaMemcpyToArray(texturelr, 0, 0, temp, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpyToArray(temp2, 0, 0, l, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
	calculateTrueDisparity(
	    temp,
	    texturelr,
	    temp2,
	    imageW,
	    imageH
        );
}
else if(j==2){
	checkCudaErrors(cudaMemcpyToArray(texturelr, 0, 0, temp, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpyToArray(temp2, 0, 0, l, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
	floatrescale(
	    temp,
	    texturelr,
	    temp2,
        3,
	    imageW,
	    imageH
        );
}
}

           
else{
       checkCudaErrors(cudaMemcpy(direction[i][j], l, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));
}


       }
    

	checkCudaErrors(cudaDeviceSynchronize());

    }

if(usingMoreGPUMemory==1){
    checkCudaErrors(cudaMemcpyToArray(texturel, 0, 0, directionl, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToArray(texturer, 0, 0, directionr, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, directionu, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToArray(compare, 0, 0, directiond, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToArray(texturelr, 0, 0, directionc, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
}
else{
if(channels==3){
    for(int i=0; i<5; i++){
	checkCudaErrors(cudaMemcpyToArray(texturel, 0, 0, direction[i][0], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToArray(texturer, 0, 0, direction[i][1], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToArray(texturelr, 0, 0, direction[i][2], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));

	calculateMeanCorr(
	    c,
	    texturel,
	    texturer,
	    texturelr,
	    imageW,
	    imageH
        );

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(direction[i][0], c, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));
    }
}
    checkCudaErrors(cudaMemcpyToArray(texturel, 0, 0, direction[0][0], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToArray(texturer, 0, 0, direction[1][0], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, direction[2][0], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToArray(compare, 0, 0, direction[3][0], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToArray(texturelr, 0, 0, direction[4][0], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
}

//for(int j=0; j<5; j++){
//for(int i=0; i<imageH; i++){
//for(int k=0; k<imageW; k++){printf("%f\t",mean[j][i*imageW+k]);} printf("\n");}printf("\n");}

//if(iter==1){
if(m==1){
cudaMemGetInfo(&freeMem, &totalMem);
printf("Free Memory:%luMB = %luKB,\tTotal Memory:%luMB = %luKB\n",freeMem/1024/1024,freeMem/1024,totalMem/1024/1024,totalMem/1024);
}
	calculatePolyDisparity(
	    l,
	    r,
	    texturel,
	    texturer,
	    texturelr,
	    imageW,
	    imageH,
	    threshold
        );
//cudaMemGetInfo(&freeMem, &totalMem);
//printf("Free Memory:%luMB = %luKB,\tTotal Memory:%luMB = %luKB\n",freeMem/1024/1024,freeMem/1024,totalMem/1024/1024,totalMem/1024);
	checkCudaErrors(cudaDeviceSynchronize());

	calculatePolyDisparity(
	    u,
	    c,
	    a_Src,
	    compare,
	    texturelr,
	    imageW,
	    imageH,
	    threshold
        );

	checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpyToArray(texturel, 0, 0, r, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToArray(texturer, 0, 0, c, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));

	compCorrelation(
	    c,
	    texturel,
	    texturer,
	    imageW,
	    imageH
	);

	checkCudaErrors(cudaDeviceSynchronize());


    checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, l, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToArray(compare, 0, 0, u, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaDeviceSynchronize());


	scaleDisparity(
	    l,
	    a_Src,
	    thresholdtest,
	    imageW,
	    imageH
        );

	scaleDisparity(
	    u,
	    compare,
	    thresholdtest,
	    imageW,
	    imageH
        );

	checkCudaErrors(cudaDeviceSynchronize());



    checkCudaErrors(cudaMemcpyToArray(texturel, 0, 0, l, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToArray(texturer, 0, 0, u, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
	if(usingMoreGPUMemory==1){
    checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, disp0, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToArray(compare, 0, 0, disp1, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaDeviceSynchronize());
	}
	else{
    checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, disparity[0], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToArray(compare, 0, 0, disparity[1], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
	}
	calculateTrueDisparity(
	    l,
	    texturel,
	    a_Src,
	    imageW,
	    imageH
        );

	calculateTrueDisparity(
	    u,
	    texturer,
	    compare,
	    imageW,
	    imageH
        );

//if((level==0)&&(iter==1)){//printf("level=%d\n",level);
if((level==0)&&(m==1)){
//	checkCudaErrors(cudaMemcpy(c, disparity[2], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
}
else{
//if(~(level==MAX_LEVEL-1)){
//if(~(level==MAX_LEVEL-1)&&(m==1)){
   checkCudaErrors(cudaMemcpyToArray(texturelr, 0, 0, c, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
	if(usingMoreGPUMemory==1){
	if(m==1){    checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, disparity[2], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaDeviceSynchronize());}
	else{
    checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, disp2, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaDeviceSynchronize());}
	}
	else{
    checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, disparity[2], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
	}
	calculateTrueConfidence(
	    c,
	    texturelr,
	    a_Src,
	    imageW,
	    imageH
	);
}
	checkCudaErrors(cudaDeviceSynchronize());

    sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
    

if(level<(MAX_LEVEL)){
realSmoothtime=smoothtime;
if(level>11){realSmoothtime=10;}
for(int j=0;j<realSmoothtime;j++){

    checkCudaErrors(cudaMemcpyToArray(texturel, 0, 0, l, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToArray(texturer, 0, 0, u, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToArray(texturelr, 0, 0, c, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaDeviceSynchronize());
        smooth(
	    l,
            texturel,
            texturelr,
            imageW,
            imageH
        );
        smooth(
	    u,
            texturer,
            texturelr,
            imageW,
            imageH
        );
        smooth(
	    c,
            texturelr,
            texturelr,
            imageW,
            imageH
        );
    checkCudaErrors(cudaDeviceSynchronize());

}}

    sdkStopTimer(&hTimer);
	gpuTime = 0.001 * sdkGetTimerValue(&hTimer);
	excutionTime[17] = gpuTime + excutionTime[17];
    
            if(m%2==0){
                if((mi/2-m/2)<7){ threshold=((mi/2-m/2)-1)*((1-0.1)/(mi/2-1.0))+0.1; }
                else{threshold=1.0;}//printf("step=%f\n",step);
	    }



/*
    checkCudaErrors(cudaMemcpyToArray(texturel, 0, 0, l, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToArray(texturer, 0, 0, u, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToArray(texturelr, 0, 0, c, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));

        convolutionRowsGPUTa(
            l,
            texturel,
            imageW,
            imageH
        );
        convolutionRowsGPUTa(
            u,
            texturer,
            imageW,
            imageH
        );
        convolutionRowsGPUTa(
            c,
            texturelr,
            imageW,
            imageH
        );

    checkCudaErrors(cudaMemcpyToArray(texturel, 0, 0, l, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToArray(texturer, 0, 0, u, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToArray(texturelr, 0, 0, c, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));

        convolutionColumnsGPUTa(
            l,
            texturel,
            imageW,
            imageH
        );
        convolutionColumnsGPUTa(
            u,
            texturer,
            imageW,
            imageH
        );
        convolutionColumnsGPUTa(
            c,
            texturelr,
            imageW,
            imageH
        );
*/
    sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
    
//	checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpyToArray(texturel, 0, 0, l, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
        convolutionRowsGPUTa(
            r,
            texturel,
            imageW,
            imageH
        );
//	checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpyToArray(texturer, 0, 0, r, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
//	checkCudaErrors(cudaDeviceSynchronize());
        convolutionColumnsGPUTa(
            l,
            texturer,
            imageW,
            imageH
        );
//	checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpyToArray(texturel, 0, 0, u, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
        convolutionRowsGPUTa(
            r,
            texturel,
            imageW,
            imageH
        );
//	checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpyToArray(texturer, 0, 0, r, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
//	checkCudaErrors(cudaDeviceSynchronize());
        convolutionColumnsGPUTa(
            u,
            texturer,
            imageW,
            imageH
        );
//	checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpyToArray(texturel, 0, 0, c, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
        convolutionRowsGPUTa(
            r,
            texturel,
            imageW,
            imageH
        );
//	checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpyToArray(texturer, 0, 0, r, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
//	checkCudaErrors(cudaDeviceSynchronize());
        convolutionColumnsGPUTa(
            c,
            texturer,
            imageW,
            imageH
        );
//	checkCudaErrors(cudaDeviceSynchronize());

    
    sdkStopTimer(&hTimer);
	gpuTime = 0.001 * sdkGetTimerValue(&hTimer);
	excutionTime[18] = gpuTime + excutionTime[18];
    
	if(usingMoreGPUMemory==1){
	checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(disp0, l, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(disp1, u, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(disp2, c, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaDeviceSynchronize());
	}
	else{
    checkCudaErrors(cudaMemcpy(disparity[0], l, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(disparity[1], u, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(disparity[2], c, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));
//    disparity=convolutionGPUTa(disparity, 3, imageW, imageH);
	}

}

if(usingMoreGPUMemory==1){
    checkCudaErrors(cudaMemcpy(disparity[0], l, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(disparity[1], u, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(disparity[2], c, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));
}

/*
for(int j=0; j<channels; j++){
for(int i=0; i<imageH; i++){
for(int k=0; k<imageW; k++){printf("%f\t",outim[j][i*imageW+k]);} printf("\n");}printf("\n");}
*/


    cudaFreeArray(a_Src);
    cudaFreeArray(compare);
    cudaFreeArray(texturel);
    cudaFreeArray(texturer);
    cudaFreeArray(texturelr);
    cudaFree(l);
    cudaFree(r);
    cudaFree(u);
    cudaFree(d);
    cudaFree(c);
if(usingMoreGPUMemory==1){
    cudaFree(directionl);
    cudaFree(directionr);
    cudaFree(directionu);
    cudaFree(directiond);
    cudaFree(directionc);
    cudaFreeArray(temp2);
    cudaFree(disp0);
    cudaFree(disp1);
    cudaFree(disp2);
    cudaFree(imleft2);
    cudaFree(imright2);
    cudaFreeArray(imleft);
    cudaFreeArray(imright);
}
else{
for(int i=0;i<5;i++){
    for (int j=0; j<channels; j++) {
//	free(direction[i][j]);
	cudaFreeHost(direction[i][j]);
    }
    free(direction[i]);
}
free(direction);
}


sdkDeleteTimer(&hTimer); // valgrind
    return disparity;

}


float ** MatchGPULib::convolutionGPUTa(float** im, int channels, int imageW, int imageH)
{
    float
    *d_Input,
    *d_Output,
    *d_Buffer;

    cudaArray
    *a_Src;

    cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();


//    float** outim;
    int  tempW, tempH;
    double gpuTime;

    StopWatchInterface *hTimer = NULL;
    sdkCreateTimer(&hTimer);
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
/*    outim = (float**)malloc(channels * sizeof(float*));
    for(int i=0; i<channels; i++){
	outim[i] = (float*)malloc(imageW * imageH * sizeof(float));
    } 
*/
    tempW=iDivUp(imageW, COLUMNS_BLOCKDIM_X)*COLUMNS_BLOCKDIM_X;
    tempH=iDivUp(imageH, (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y))*(COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y);

    checkCudaErrors(cudaMalloc((void **)&d_Input,   imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_Output,  imageW * imageH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_Buffer , imageW * imageH * sizeof(float)));

    checkCudaErrors(cudaMallocArray(&a_Src, &floatTex, imageW, imageH));


	sdkStopTimer(&hTimer);
	gpuTime = 0.001 * sdkGetTimerValue(&hTimer);

    for(int j=0;j<channels;j++) 
    {
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
    checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, im[j], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&hTimer);
	gpuTime = 0.001 * sdkGetTimerValue(&hTimer);
	excutionTime[8] = gpuTime + excutionTime[8];

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);

        convolutionRowsGPUTa(
            d_Output,
            a_Src,
            imageW,
            imageH
        );
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

        convolutionColumnsGPUTa(
            d_Output,
            a_Src,
            imageW,
            imageH
        );

	checkCudaErrors(cudaDeviceSynchronize());
	sdkStopTimer(&hTimer);
	gpuTime = 0.001 * sdkGetTimerValue(&hTimer);
	excutionTime[7] = gpuTime + excutionTime[7];

	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	checkCudaErrors(cudaMemcpy(im[j], d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));
	sdkStopTimer(&hTimer);
	gpuTime = 0.001 * sdkGetTimerValue(&hTimer);
	excutionTime[8] = gpuTime + excutionTime[8];
    }

	sdkResetTimer(&hTimer);

    checkCudaErrors(cudaFree(d_Buffer));
    checkCudaErrors(cudaFree(d_Output));
    checkCudaErrors(cudaFree(d_Input));
    cudaFreeArray(a_Src);

sdkDeleteTimer(&hTimer); // valgrind
    return im;
	
}



float ** MatchGPULib::hierarchicalDisparity(float** im, float*** foveated, int channels, int widthInit, int heightInit)
{
    int level,disp,top,i;
    int x,y,xs,ys,xl,yl,l1,r1,l2,r2,u,d,planes,rows,cols,rows1,cols1,foveatedrows,foveatedcols,temp,temp2,temp3;
    double levelscale;
    float **disparity, **pAlias1, ***pAlias2;
    int height[MAX_LEVEL-1], width[MAX_LEVEL-1];
    float *d_Output;
    int imageW,imageH,imageW2,imageH2;

    char string[20];
    IplImage *outImg;

    cudaArray
    *a_Src;

    cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();


    width[0] = widthInit;
    height[0] = heightInit;
    for(int i=0; i<=foveatelevel; i++)
    {
	width[i+1]=width[i]/SCALE;
	height[i+1]=height[i]/SCALE;
    }

    disparity = (float**)malloc(channels * sizeof(float*));
    for(int i=0; i<channels; i++){
        disparity[i] = (float*)malloc(heightInit * widthInit * sizeof(float));
    }
    pAlias2 = (float***)malloc(foveatelevel * sizeof(float**));
    for(int i=0; i<foveatelevel; i++)
    {
        pAlias2[i] = (float**)malloc(channels * sizeof(float*));
        for(int k=0;k<channels;k++){
            pAlias2[i][k] = (float*)malloc(width[i] * height[i] * sizeof(float));
        }
    }

    xs=width[foveatelevel-1]/2;
    ys=height[foveatelevel-1]/2;

    foveatedcols=height[foveatelevel-1];
    foveatedrows=width[foveatelevel-1];

    for(level=foveatelevel-1;level>0;level--)
    {

        levelscale = SCALE;
        rows=width[level-1];
        cols=height[level-1];
        rows1 = width[level];
        cols1 = height[level];
        x=rows/2;
        y=cols/2;

        l1=x-xs;
        u=y-ys;

        if(level==foveatelevel-1){pAlias1 = foveated[foveatelevel-1];}


        imageW=rows1;//foveatedrows;
        imageH=cols1;//foveatedcols;
        imageW2=rows;
        imageH2=cols;

        checkCudaErrors(cudaMalloc((void **)&d_Output,  imageW2 * imageH2 * sizeof(float)));
        checkCudaErrors(cudaMallocArray(&a_Src, &floatTex, imageW, imageH));
        for(int j=0; j<channels; j++){
            checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, 
                pAlias1[j], imageW * imageH * sizeof(float), 
                cudaMemcpyHostToDevice));
            checkCudaErrors(cudaDeviceSynchronize());
            partsubsampleDispGPU(d_Output, a_Src, imageW, imageH,
	            levelscale, imageW2, imageH2);
            checkCudaErrors(cudaDeviceSynchronize());
            checkCudaErrors(cudaMemcpy(pAlias2[level-1][j], 
                d_Output, imageW2 * imageH2 * sizeof(float), 
                cudaMemcpyDeviceToHost));
        }
        cudaFreeArray(a_Src);
        cudaFree(d_Output);

        pAlias1 = pAlias2[level-1];

        for(int j=0; j<channels; j++)
            for(int i=0;i<foveatedcols;i++){
                memcpy(&pAlias1[j][(u+i)*width[level-1]+l1],
                     &foveated[level-1][j][i*foveatedrows],
                      foveatedrows * sizeof(float));
            }

    }   //END of for(level=foveatelevel-1;level>0;level--)
    
    disparity = pAlias2[0];

/* valgrind
for(int i=0; i<foveatelevel; i++){
 for(int k=0;k<channels;k++){
        free(pAlias2[i][k]);
}
free(pAlias2[i]);
    }
free(pAlias2);
*/
cvReleaseImage(&outImg); // valgrind
    return disparity;

}



