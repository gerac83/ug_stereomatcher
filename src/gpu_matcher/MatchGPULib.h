#ifndef _MATCHGPULIB_H
#define _MATCHGPULIB_H

#include <cv_bridge/cv_bridge.h>

class MatchGPULib
{
	public:
	bool  foveatedmatching;
	int  foveatelevel;
	int fovH;
	int fovW;
	MatchGPULib(int argc, char **argv);
	
	int getFoveaWidth();
	int getFoveaHeight();
	int getFoveateLevel();

/* setting Fovea parameters */
void setFoveaWidth(int rows);
void setFoveaHeight(int cols); 
void setFoveated(int fov);
     
        float **match(cv_bridge::CvImagePtr cv_ptrL, cv_bridge::CvImagePtr cv_ptrR, int fov);
        int initStack(cv_bridge::CvImagePtr cv_ptrL, cv_bridge::CvImagePtr cv_ptrR);
	float ***matchStack(cv_bridge::CvImagePtr cv_ptrL, cv_bridge::CvImagePtr cv_ptrR); /* No longer in use for foveated stack retreival - we use 'matchStackPyramid' instead */      
        float ***matchStackPyramid(cv_bridge::CvImagePtr cv_ptrL, cv_bridge::CvImagePtr cv_ptrR, float ***leftFoveated, float ***rightFoveated);
	int iDivUp(int a, int b);
	float gaussian(float x);
	void gaussiankernel(float* k);
	float **convolutionCPU(float** im, float* h_Kernel, int channels, int imageW, int imageH);
	float **subsampledimageCPU(float**im, float scalefactor, int channels, int imageW, int imageH, int imageW2, int imageH2);
	float **convolutionGPU(float** im, int channels, int imageW, int imageH);
	float **subsampledimageGPU(float**im, float scalefactor, int channels, int imageW, int imageH, int imageW2, int imageH2);
	float ***CreatePyramidFromImage(float **im, int channels, int heightInit, int widthInit, float *h_Kernel);
	float ***CreateFoveatedPyramid(float ***im, int channels, int heightInit, int widthInit);
	float ***matching(float ***iml, float ***imr, int channels, int heightInit, int widthInit);
	int differenceIterations(float*DH,float*DV,float*conf,float*OldDH,float*OldDV,float threshold, int imageW, int imageH);
	float weightedDifference(float*D,float*OldD,float*conf, int imageW, int imageH);
	float **warpRightImage(float **right, float **disparity, int channels, int imageW, int imageH);
	float **subsampleDisp(float**im, float scalefactor, int channels, int imageW, int imageH, int imageW2, int imageH2);
	float **foveatedsubsampleDisp(float**im, float scalefactor, int channels, int imageW, int imageH, int imageW2, int imageH2);
	float **matchlevel(float **left, float **right, float **disparity, int channels, int imageW, int imageH, int level, int iter, float threshold);
	float **convolutionGPUTa(float** im, int channels, int imageW, int imageH);
	float **hierarchicalDisparity(float** im, float*** foveated, int channels, int widthInit, int heightInit);
	
};

#endif
