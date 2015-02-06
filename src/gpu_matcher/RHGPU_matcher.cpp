
#include <ros/ros.h>
#include <ros/package.h>

#include <ros/time.h>

#include <iostream>
#include <sstream>
#include <time.h>
#include <stdlib.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <vector>
#include <errno.h>
#include <fstream>
#include <math.h>


#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/subscriber.h>
#include <sensor_msgs/JointState.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <stereo_msgs/DisparityImage.h> 
#include <ug_stereomatcher/foveatedstack.h>

#include <cv_bridge/cv_bridge.h>

#include <tf/transform_listener.h>
#include <std_msgs/String.h>
#include "MatchGPULib.h"
#include <ug_stereomatcher/GetDisparitiesGPU.h>

namespace enc = sensor_msgs::image_encodings;
using namespace cv;
using namespace std;
//#define malloc(n) GC_malloc(n)
// Messages
static const char CAM_SUB_LEFT[] = "input_left_image";
static const char CAM_SUB_RIGHT[] = "input_right_image";
static const char CAM_PUB_HOR[] = "output_disparityH";
static const char CAM_PUB_VER[] = "output_disparityV";
static const char CAM_PUB_CONF[] = "output_disparityC"; // Confidence map
static const char CAM_PUB_STACK_HOR[] = "output_stackH";
static const char CAM_PUB_STACK_VER[] = "output_stackV";
static const char CAM_PUB_STACK_CONF[] = "output_stackC";  // Confidence map
static const char CAM_PUB_STACK_LEFTP[] = "output_stackL_pyramid"; // left image from pyramid
static const char CAM_PUB_STACK_LEFTR[] = "output_stackR_pyramid"; // right image from pyramid
static const char DISPARITIES_SRV[] = "get_disparities_srv";

// Parameter server
static const char FOVEATEDQ[] = "foveated";

int nrFramesL;
int nrFramesR;

class GPU_matcher 
{
public:
    vector<int> compression_params;
    int foveated;
    int cmd_argc;
    char **cmd_argv;
    ros::Publisher stereo_diH_pub;
    ros::Publisher stereo_diV_pub;
    ros::Publisher stereo_diC_pub;

    ros::Publisher stack_diH_pub;
    ros::Publisher stack_diV_pub;
    ros::Publisher stack_diC_pub;

    ros::Publisher stack_diL_pub;
    ros::Publisher stack_diR_pub;

    ros::ServiceServer disparities_srv_; /* manages a service */

    GPU_matcher(int argc, char **argv):
        it_(nh_), /* nh_ is a node handler*/
        imL_sub_(it_, CAM_SUB_LEFT, 1), /* 'imL_sub' of type  ImageSubscriber and 'CAM_SUB_LEFT' is the msg*/
        imR_sub_(it_, CAM_SUB_RIGHT, 1),  /* of type  ImageSubscriber*/
        sync(syncPolicy(1), imL_sub_, imR_sub_)//, sub_)
    {

        cmd_argc=argc;
        cmd_argv=argv;
        
        if (nh_.hasParam(FOVEATEDQ)){
            nh_.getParam(FOVEATEDQ, foveated);
        }
        else{
            ROS_WARN("foveated option has not been set. Matcher is on non-foveated mode!");
            foveated = 0; // 0 : non foveated , 1 : foveated
        }

        stack_diH_pub = nh_.advertise<ug_stereomatcher::foveatedstack>(CAM_PUB_STACK_HOR, 1);
        stack_diV_pub = nh_.advertise<ug_stereomatcher::foveatedstack>(CAM_PUB_STACK_VER, 1);
        stack_diC_pub = nh_.advertise<ug_stereomatcher::foveatedstack>(CAM_PUB_STACK_CONF, 1);

        stack_diL_pub = nh_.advertise<ug_stereomatcher::foveatedstack>(CAM_PUB_STACK_LEFTP, 1); /* not needed */
        stack_diR_pub = nh_.advertise<ug_stereomatcher::foveatedstack>(CAM_PUB_STACK_LEFTR, 1); /* not needed */

        stereo_diH_pub = nh_.advertise<stereo_msgs::DisparityImage>(CAM_PUB_HOR, 1);
        stereo_diV_pub = nh_.advertise<stereo_msgs::DisparityImage>(CAM_PUB_VER, 1);
        stereo_diC_pub = nh_.advertise<stereo_msgs::DisparityImage>(CAM_PUB_CONF, 1); /* confidence map */


        disparities_srv_ = nh_.advertiseService(DISPARITIES_SRV, &GPU_matcher::disparitySrv, this);
        sync.registerCallback(boost::bind(&GPU_matcher::mainRoutine, this, _1, _2));
        ROS_INFO("GPUMatcher node initialised!!");
    }

    ~GPU_matcher()
    {
        ROS_INFO("Bye!");
    }
    
    void mainRoutine(const sensor_msgs::ImageConstPtr& imL, const sensor_msgs::ImageConstPtr& imR)
    {
        ROS_INFO("Received disparities!");
        float **finDisp, **dataH, **dataV, **dataC, **dataL, **dataR;
        float *dataStackH, *dataStackV;
        float ***stackDisp, ***leftFov , ***rightFov;
        cv_bridge::CvImagePtr cv_ptrL, cv_ptrR;

        stereo_msgs::DisparityImagePtr stereo_diH(new stereo_msgs::DisparityImage()), stereo_diV(new stereo_msgs::DisparityImage()), stereo_diC(new stereo_msgs::DisparityImage());
        ug_stereomatcher::foveatedstack fovStackH, fovStackV, fovStackC, fovStackL, fovStackR;

        cv_bridge::CvImagePtr cv_imH(new cv_bridge::CvImage()), cv_imV(new cv_bridge::CvImage()), cv_imC(new cv_bridge::CvImage()), cv_imL(new cv_bridge::CvImage()), cv_imR(new cv_bridge::CvImage());

        ROS_INFO("Reading messages!");
        try
        {
            // Get images
            cv_ptrL = cv_bridge::toCvCopy(imL, enc::RGB8);
            cv_ptrR = cv_bridge::toCvCopy(imR, enc::RGB8);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("Could not convert from '%s' to 'rgb8'.", imL->encoding.c_str());
            return;
        }
        
        if (nh_.hasParam(FOVEATEDQ)){
            nh_.getParam(FOVEATEDQ, foveated);
        }
        else{
            ROS_WARN("foveated option has not been set. Matcher is on non-foveated mode!");
            foveated = 0; // 0 : non foveated , 1 : foveated
        }

        MatchGPULib mgpu(cmd_argc,cmd_argv);
        mgpu.setFoveated(foveated); /* Sets the fovea flag (0 or 1)*/
        
        if (foveated == 1){

            ros::WallTime start = ros::WallTime::now();
            mgpu.initStack(cv_ptrL, cv_ptrR);  //returns the size of the fovea region

            /* allocate mem for left and right fovea from pyramid */
            leftFov = (float***)malloc(14 * sizeof(float**)); /* 7 is the number fovea levels , 14 is max level in pyramid creation */
            rightFov = (float***)malloc(14 * sizeof(float**)); /* 7 is the number fovea levels , 14 is max level in pyramid creation */

            for(int level=0;level<14;level++){
                leftFov[level] = (float**)malloc(3 * sizeof(float*));
                rightFov[level] = (float**)malloc(3 * sizeof(float*));

                for(int k=0;k<3;k++){
                    leftFov[level][k] = (float*)malloc(mgpu.getFoveaWidth() * mgpu.getFoveaHeight()* sizeof(float));
                    rightFov[level][k] = (float*)malloc(mgpu.getFoveaWidth() * mgpu.getFoveaHeight()* sizeof(float));
                }}

            stackDisp = mgpu.matchStackPyramid(cv_ptrL, cv_ptrR, leftFov, rightFov);

            ros::WallTime end = ros::WallTime::now();
            ros::WallDuration dur = end - start;
            ROS_INFO("Foveated Disparity took %f Seconds", dur.toSec());

            ROS_INFO_STREAM("fov Images matched!");

            dataL = (float**)malloc(mgpu.getFoveaHeight() *
                                    mgpu.getFoveateLevel() * 3 * sizeof(float*));
            dataR = (float**)malloc(mgpu.getFoveaHeight() *
                                    mgpu.getFoveateLevel() * 3 * sizeof(float*));

            for(int i=0; i < mgpu.getFoveaHeight() *
                mgpu.getFoveateLevel() * 3; i++){
                dataL[i] = (float*)malloc(mgpu.getFoveaWidth() *
                                          sizeof(float));
                dataR[i] = (float*)malloc(mgpu.getFoveaWidth() *
                                          sizeof(float));
            }

            /* we store the stack of right and left images from pyramid - with seperates channels for each image in the pyramid level */
            for (int k=0; k < mgpu.getFoveateLevel(); k++)
                for(int ch=0; ch<3; ch++)
                    for(int i=0; i<mgpu.getFoveaHeight(); i++)
                        for(int j=0; j<mgpu.getFoveaWidth(); j++){

                            (dataL[k*(mgpu.getFoveaHeight()*3)+ ch*mgpu.getFoveaHeight() +i][j]) =
                                    (float)(leftFov[k][ch][i*(mgpu.getFoveaWidth())+j]);
                            (dataR[k*(mgpu.getFoveaHeight()*3)+ ch*mgpu.getFoveaHeight() +i][j]) =
                                    (float)(rightFov[k][ch][i*(mgpu.getFoveaWidth())+j]);

                        }
            /*(dataL[k*(mgpu.getFoveaHeight()) +i][j]) = (float)(leftFov[k][0][i*(mgpu.getFoveaWidth())+j]);*/ /* if you only want one channel, replace 'ch' with 0,1,2*/
            /*(dataR[k*(mgpu.getFoveaHeight()) +i][j]) = (float)(rightFov[k][0][i*(mgpu.getFoveaWidth())+j]);*/ /* if you only want one channel, replace 'ch' with 0,1,2*/

            Mat ch4 = Mat::zeros(mgpu.getFoveateLevel() * mgpu.getFoveaHeight() * 3, mgpu.getFoveaWidth(), CV_32F);
            Mat ch5 = Mat::zeros(mgpu.getFoveateLevel() * mgpu.getFoveaHeight() * 3, mgpu.getFoveaWidth(), CV_32F);

            for(int i = 0; i < mgpu.getFoveateLevel() *mgpu.getFoveaHeight()*3 ; i++)
                for(int j = 0; j < mgpu.getFoveaWidth() ; j++)
                {
                    ch4.at<float>(i,j) = dataL[i][j];
                    ch5.at<float>(i,j) = dataR[i][j];

                }

            cv_imL->image = ch4.clone();
            cv_imR->image = ch5.clone();

            ROS_INFO_STREAM("cols : " << mgpu.getFoveaHeight());
            ROS_INFO_STREAM("rows : " << mgpu.getFoveaWidth());
            ROS_INFO_STREAM("Fovlevel : " << mgpu.getFoveateLevel());
            ROS_INFO_STREAM("Size FLeft: " << cv_imL->image.size());
            ROS_INFO_STREAM("Size RLeft: " << cv_imR->image.size());

            cv_imL->header = cv_ptrL->header;
            cv_imR->header = cv_ptrR->header;

            // message original image size
            fovStackL.im_width = cv_ptrL->image.cols;
            fovStackL.im_height = cv_ptrL->image.rows;
            fovStackR.im_width = cv_ptrR->image.cols;
            fovStackR.im_height = cv_ptrR->image.rows;

            // message stack sizes
            fovStackL.roi_width = mgpu.getFoveaWidth();
            fovStackL.roi_height = mgpu.getFoveaHeight();
            fovStackL.num_levels = mgpu.getFoveateLevel();

            fovStackR.roi_width = mgpu.getFoveaWidth();
            fovStackR.roi_height = mgpu.getFoveaHeight();
            fovStackR.num_levels = mgpu.getFoveateLevel();

            cv_imL->encoding = sensor_msgs::image_encodings::TYPE_32FC1;
            cv_imR->encoding = sensor_msgs::image_encodings::TYPE_32FC1;

            fovStackL.image_stack = *(cv_imL->toImageMsg());
            fovStackR.image_stack = *(cv_imR->toImageMsg());

            // message header
            fovStackL.header = cv_ptrL->header;
            fovStackR.header = cv_ptrR->header;

            stack_diL_pub.publish(fovStackL);
            stack_diR_pub.publish(fovStackR);


            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            /* The 'matchStack function was replaced with 'matchStackPyramid'*/
            //stackDisp = mgpu.matchStack(cv_ptrL,cv_ptrR);
            //ROS_INFO("fov Images matched!");
            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////


            dataH = (float**)malloc(mgpu.getFoveaHeight() *
                                    mgpu.getFoveateLevel() * sizeof(float*));
            dataV = (float**)malloc(mgpu.getFoveaHeight() *
                                    mgpu.getFoveateLevel()* sizeof(float*));
            dataC = (float**)malloc(mgpu.getFoveaHeight() *
                                    mgpu.getFoveateLevel()* sizeof(float*));

            for(int i=0; i < mgpu.getFoveaHeight() *
                mgpu.getFoveateLevel(); i++){
                dataH[i] = (float*)malloc(mgpu.getFoveaWidth() *
                                          sizeof(float));
                dataV[i] = (float*)malloc(mgpu.getFoveaWidth() *
                                          sizeof(float));
                dataC[i] = (float*)malloc(mgpu.getFoveaWidth() *
                                          sizeof(float));
            }

            for (int k=0; k < mgpu.getFoveateLevel(); k++)
                for(int i=0; i<mgpu.getFoveaHeight(); i++)
                    for(int j=0; j<mgpu.getFoveaWidth(); j++){

                        (dataH[k*(mgpu.getFoveaHeight())+i][j]) =
                                (float)(stackDisp[k][0][i*(mgpu.getFoveaWidth())+j]);
                        (dataV[k*(mgpu.getFoveaHeight())+i][j]) =
                                (float)(stackDisp[k][1][i*(mgpu.getFoveaWidth())+j]);
                        (dataC[k*(mgpu.getFoveaHeight())+i][j]) =
                                (float)(stackDisp[k][2][i*(mgpu.getFoveaWidth())+j]);
                    }

            Mat ch1 = Mat::zeros(mgpu.getFoveateLevel() * mgpu.getFoveaHeight() , mgpu.getFoveaWidth(), CV_32F);
            Mat ch2 = Mat::zeros(mgpu.getFoveateLevel() * mgpu.getFoveaHeight() , mgpu.getFoveaWidth(), CV_32F);
            Mat ch3 = Mat::zeros(mgpu.getFoveateLevel() * mgpu.getFoveaHeight() , mgpu.getFoveaWidth(), CV_32F);

            for(int i = 0; i < mgpu.getFoveateLevel() *mgpu.getFoveaHeight() ; i++)
                for(int j = 0; j < mgpu.getFoveaWidth() ; j++)
                {
                    ch1.at<float>(i,j) = dataH[i][j];
                    ch2.at<float>(i,j) = dataV[i][j];
                    ch3.at<float>(i,j) = dataC[i][j];
                }


            cv_imH->image = ch1.clone();
            cv_imV->image = ch2.clone();
            cv_imC->image = ch3.clone();

            ROS_INFO_STREAM("Size Fhoriz: " << cv_imH->image.size());
            ROS_INFO_STREAM("Size Fvert: " << cv_imV->image.size());
            ROS_INFO_STREAM("Size Fconfidence map: " << cv_imC->image.size());

            // message header
            cv_imH->header = cv_ptrL->header;
            cv_imV->header = cv_ptrR->header;
            cv_imC->header = cv_ptrL->header;


            // message original image size
            fovStackH.im_width = cv_ptrL->image.cols;
            fovStackH.im_height = cv_ptrL->image.rows;
            fovStackV.im_width = cv_ptrL->image.cols;
            fovStackV.im_height = cv_ptrL->image.rows;
            fovStackC.im_width = cv_ptrL->image.cols;
            fovStackC.im_height = cv_ptrL->image.rows;

            // message stack sizes
            fovStackH.roi_width = mgpu.getFoveaWidth();
            fovStackH.roi_height = mgpu.getFoveaHeight();
            fovStackH.num_levels = mgpu.getFoveateLevel();
            fovStackV.roi_width = mgpu.getFoveaWidth();
            fovStackV.roi_height = mgpu.getFoveaHeight();
            fovStackV.num_levels = mgpu.getFoveateLevel();
            fovStackC.roi_width = mgpu.getFoveaWidth();
            fovStackC.roi_height = mgpu.getFoveaHeight();
            fovStackC.num_levels = mgpu.getFoveateLevel();

            cv_imH->encoding = sensor_msgs::image_encodings::TYPE_32FC1;
            cv_imV->encoding = sensor_msgs::image_encodings::TYPE_32FC1;
            cv_imC->encoding = sensor_msgs::image_encodings::TYPE_32FC1;


            fovStackH.image_stack = *(cv_imH->toImageMsg());
            fovStackV.image_stack = *(cv_imV->toImageMsg());
            fovStackC.image_stack = *(cv_imC->toImageMsg());

            // message header
            fovStackH.header = cv_ptrL->header;
            fovStackV.header = cv_ptrR->header;
            fovStackC.header = cv_ptrL->header;

            stack_diH_pub.publish(fovStackH);
            stack_diV_pub.publish(fovStackV);
            stack_diC_pub.publish(fovStackC);

            ROS_INFO("Foveated Stack published!");

            //saveImages("FHoriz.png","FVert.png",cv_imH->image, cv_imV->image, 1); /* foveated Horiz and Vert*/
            //saveImages("FLeft.png","FRight.png",cv_imL->image, cv_imR->image, 1); /* foveated Left and right image from Pyramid, for confidence map : replace it with 'cv_imC->image' */
            

            /* write to the file for Profiling
            FILE * fp;
            fp = fopen("FHDisp.txt","w");
            for(int i = 0; i < mgpu.getFoveateLevel() *mgpu.getFoveaHeight() ; i++){
              for(int j = 0; j < mgpu.getFoveaWidth() ; j++)
                            {fprintf(fp,"%f\t",dataH[i][j]);}fprintf(fp,"\n");}fclose(fp);
            fp = fopen("FVDisp.txt","w");
            for(int i = 0; i < mgpu.getFoveateLevel() *mgpu.getFoveaHeight() ; i++){
              for(int j = 0; j < mgpu.getFoveaWidth() ; j++)
                            {fprintf(fp,"%f\t",dataV[i][j]);}fprintf(fp,"\n");}fclose(fp);
            fp = fopen("FCDisp.txt","w");
            for(int i = 0; i < mgpu.getFoveateLevel() *mgpu.getFoveaHeight() ; i++){
              for(int j = 0; j < mgpu.getFoveaWidth() ; j++)
                            {fprintf(fp,"%f\t",dataC[i][j]);}fprintf(fp,"\n");}fclose(fp);
            */

            // free mem
            for(int level=0;level<14;level++){
                for (int i=0; i < 3; i++){
                    free(leftFov[level][i]);
                    free(rightFov[level][i]);
                }
                free(leftFov[level]); free(rightFov[level]);}
            free(leftFov);free(rightFov);

            for(int i=0; i<mgpu.getFoveaHeight() *mgpu.getFoveateLevel() * 3 ; i++){
                free(dataL[i]); free(dataR[i]);
            }
            free(dataL);free(dataR);

            for(int i=0; i<mgpu.getFoveaHeight() *  mgpu.getFoveateLevel(); i++){
                free(dataH[i]);
                free(dataV[i]);
                free(dataC[i]);
            }
            free(dataH);
            free(dataV);
            free(dataC);

            for (int k=0; k < mgpu.getFoveateLevel(); k++){
                for (int i=0; i < 3; i++){
                    free(stackDisp[k][i]);
                }free(stackDisp[k]);}
            free(stackDisp);


        }else{ /* this part can aslo generate foveated version, but it is not stack of images , if foveated is set to 1 */
            ros::WallTime start = ros::WallTime::now();
            finDisp = mgpu.match(cv_ptrL,cv_ptrR,foveated);
            ros::WallTime end = ros::WallTime::now();
            ros::WallDuration dur = end - start;
            ROS_INFO("Non Foveated Disparity took %f Seconds", dur.toSec());

            ROS_INFO("non fov Images matched!");

            Mat ch1 = Mat::zeros(cv_ptrL->image.rows, cv_ptrL->image.cols, CV_32F);
            Mat ch2 = Mat::zeros(cv_ptrR->image.rows, cv_ptrR->image.cols, CV_32F);

            Mat ch3 = Mat::zeros(cv_ptrL->image.rows, cv_ptrL->image.cols, CV_32F);  /*you can use R as well - another cv_bridge type to convert opencv image to ros image msg */

            for(int i = 0; i < cv_ptrL->image.rows; i++)
                for(int j = 0; j < cv_ptrL->image.cols; j++)
                {
                    ch1.at<float>(i,j) = (float)(finDisp[0][i*cv_ptrL->image.cols+j]);
                    ch2.at<float>(i,j) = (float)(finDisp[1][i*cv_ptrL->image.cols+j]);

                    ch3.at<float>(i,j) = (float)(finDisp[2][i*cv_ptrL->image.cols+j]); /* same size as left and right */
                }


            cv_imH->image = ch1.clone();
            cv_imV->image = ch2.clone();
            cv_imC->image = ch3.clone();

            ROS_INFO_STREAM("Size Horiz: " << cv_imH->image.size());
            ROS_INFO_STREAM("Size Vert: " << cv_imV->image.size());
            ROS_INFO_STREAM("Size confidence map: " << cv_imC->image.size());

            // DEBUG ********************
            //int i = 100;
            //int j = 100;
            //ROS_INFO_STREAM(dataH[i][j] << " " << dataV[i][j]);
            //ROS_INFO_STREAM(cv_imH->image.at<float>(i,j) << " " << cv_imV->image.at<float>(i,j));
            //ROS_INFO_STREAM(cv_imC->image.at<float>(i,j));
            // **************************

            cv_imH->header = cv_ptrL->header;
            cv_imV->header = cv_ptrR->header;
            cv_imC->header = cv_ptrL->header;

            cv_imH->encoding = sensor_msgs::image_encodings::TYPE_32FC1;
            cv_imV->encoding = sensor_msgs::image_encodings::TYPE_32FC1;
            cv_imC->encoding = sensor_msgs::image_encodings::TYPE_32FC1;

            stereo_diH->image = *(cv_imH->toImageMsg());
            stereo_diV->image = *(cv_imV->toImageMsg());
            stereo_diC->image = *(cv_imC->toImageMsg());

            stereo_diH->header = cv_ptrL->header;
            stereo_diV->header = cv_ptrR->header;
            stereo_diC->header = cv_ptrL->header;

            stereo_diH_pub.publish(stereo_diH); /*'sterio_diH' is a message object. We stuff it with data, and then publish it */
            stereo_diV_pub.publish(stereo_diV); /*'sterio_div' is a message object. We stuff it with data, and then publish it */
            stereo_diC_pub.publish(stereo_diC); /*'sterio_div' is a message object. We stuff it with data, and then publish it */

            ROS_INFO("non fov Disparity published!");
            //saveImages("L.png","R.png",cv_ptrL->image, cv_ptrR->image, 1); /* This is the resized version of the original image*/
            //saveImages("Horiz.png","Vert.png",cv_imH->image, cv_imV->image, 1);  /* foveated Horiz and Vert, for confidence map : replace it with 'cv_imC->image' */
            //saveImages("Horiz.png", "Conf.png",cv_imH->image, cv_imC->image, 1);

            //free mem
            for (int i =0; i<3;i++)
                free(finDisp[i]);
            free(finDisp);

        }
    

    }

    /* changes need to be made to the GetDisparitiesGPU.srv , i.e : I added the fovea ones to it as well */
    bool disparitySrv(ug_stereomatcher::GetDisparitiesGPU::Request& req, ug_stereomatcher::GetDisparitiesGPU::Response& rsp)
    {
        float **finDisp, **dataH, **dataV,**dataC;
        float *dataStackH, *dataStackV;
        float ***stackDisp;

        cv_bridge::CvImagePtr cv_ptrL, cv_ptrR ;
        stereo_msgs::DisparityImagePtr stereo_diH(new stereo_msgs::DisparityImage()), stereo_diV(new stereo_msgs::DisparityImage()), stereo_diC(new stereo_msgs::DisparityImage());
        ug_stereomatcher::foveatedstack fovStackH,fovStackV,fovStackC;

        cv_bridge::CvImagePtr cv_imH(new cv_bridge::CvImage()), cv_imV(new cv_bridge::CvImage()), cv_imC(new cv_bridge::CvImage());

        ROS_INFO("Reading messages!");
        try
        {
            // Get images
            cv_ptrL = cv_bridge::toCvCopy(req.imL, enc::RGB8);
            cv_ptrR = cv_bridge::toCvCopy(req.imR, enc::RGB8);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("Could not convert from '%s' to 'rgb8'.", req.imL.encoding.c_str());
            return false;
        }

        if (nh_.hasParam(FOVEATEDQ)){
            nh_.getParam(FOVEATEDQ, foveated);
        }
        else{
            ROS_WARN("foveated option has not been set. Matcher is on non-foveated mode!");
            foveated = 0; // 0 : non foveated , 1 : foveated
        }

        MatchGPULib mgpu(cmd_argc,cmd_argv);
        mgpu.setFoveated(foveated);
        
        if (foveated == 1){

            stackDisp = mgpu.matchStack(cv_ptrL,cv_ptrR); /* we only need disparity */
            ROS_INFO("Fov Images matched! - in SRV");


            dataH = (float**)malloc(mgpu.getFoveaHeight() *
                                    mgpu.getFoveateLevel() * sizeof(float*));
            dataV = (float**)malloc(mgpu.getFoveaHeight() *
                                    mgpu.getFoveateLevel()* sizeof(float*));
            dataC = (float**)malloc(mgpu.getFoveaHeight() *
                                    mgpu.getFoveateLevel()* sizeof(float*));

            for(int i=0; i < mgpu.getFoveaHeight() *
                mgpu.getFoveateLevel(); i++){
                dataH[i] = (float*)malloc(mgpu.getFoveaWidth() *
                                          sizeof(float));
                dataV[i] = (float*)malloc(mgpu.getFoveaWidth() *
                                          sizeof(float));
                dataC[i] = (float*)malloc(mgpu.getFoveaWidth() *
                                          sizeof(float));
            }
            for (int k=0; k < mgpu.getFoveateLevel(); k++)
                for(int i=0; i<mgpu.getFoveaHeight(); i++)
                    for(int j=0; j<mgpu.getFoveaWidth(); j++){
                        (dataH[k*mgpu.getFoveaHeight()+i][j]) =
                                (float)(stackDisp[k][0][i*mgpu.getFoveaHeight()+j]);
                        (dataV[k*mgpu.getFoveaHeight()+i][j]) =
                                (float)(stackDisp[k][1][i*mgpu.getFoveaHeight()+j]);
                        (dataC[k*mgpu.getFoveaHeight()+i][j]) =
                                (float)(stackDisp[k][2][i*mgpu.getFoveaHeight()+j]);
                    }


            Mat ch1 = Mat::zeros(mgpu.getFoveateLevel() * mgpu.getFoveaHeight() , mgpu.getFoveaWidth(), CV_32F);
            Mat ch2 = Mat::zeros(mgpu.getFoveateLevel() * mgpu.getFoveaHeight() , mgpu.getFoveaWidth(), CV_32F);
            Mat ch3 = Mat::zeros(mgpu.getFoveateLevel() * mgpu.getFoveaHeight() , mgpu.getFoveaWidth(), CV_32F);


            for(int i = 0; i < mgpu.getFoveaHeight() ; i++)
                for(int j = 0; j < mgpu.getFoveaWidth() ; j++)
                {
                    ch1.at<float>(i,j) = dataH[i][j];
                    ch2.at<float>(i,j) = dataV[i][j];
                    ch3.at<float>(i,j) = dataC[i][j];
                }

            cv_imH->image = ch1.clone();
            cv_imV->image = ch2.clone();
            cv_imC->image = ch3.clone();


            // message header
            cv_imH->header = cv_ptrL->header;
            cv_imV->header = cv_ptrR->header;
            cv_imC->header = cv_ptrL->header;

            /*
            fdispH.im_width = cv_ptrL->image.cols;
            fdispH.im_height = cv_ptrL->image.rows;
            fdispV.im_width = cv_ptrL->image.cols;
            fdispV.im_height = cv_ptrL->image.rows;
                        fdispC.im_width = cv_ptrL->image.cols;
            fdispC.im_height = cv_ptrL->image.rows;

            // message stack sizes
            fdispH.roi_width = mgpu.getFoveaWidth();
            fdispH.roi_height = mgpu.getFoveaHeight();
            fdispH.num_levels = mgpu.getFoveateLevel();
            //fdispV.roi_width = mgpu.getFoveaWidth();
            fdispV.roi_height = mgpu.getFoveaHeight();
            fdispV.num_levels = mgpu.getFoveateLevel();
                        fdispC.roi_width = mgpu.getFoveaWidth();
            fdispC.roi_height = mgpu.getFoveaHeight();
            fdispC.num_levels = mgpu.getFoveateLevel();
                        */


            cv_imH->encoding = sensor_msgs::image_encodings::TYPE_32FC1;
            cv_imV->encoding = sensor_msgs::image_encodings::TYPE_32FC1;
            cv_imC->encoding = sensor_msgs::image_encodings::TYPE_32FC1;

            rsp.fdispH.image_stack = *(cv_imH->toImageMsg());
            rsp.fdispV.image_stack = *(cv_imV->toImageMsg());
            rsp.fdispC.image_stack = *(cv_imC->toImageMsg());

            // message header
            rsp.fdispH.header = cv_ptrL->header;
            rsp.fdispV.header = cv_ptrR->header;
            rsp.fdispC.header = cv_ptrL->header;

            ROS_INFO("Foveated computed!");

            // free mem
            for(int i=0; i<mgpu.getFoveaHeight() *  mgpu.getFoveateLevel(); i++){
                free(dataH[i]);
                free(dataV[i]);
                free(dataC[i]);
            }
            free(dataH);
            free(dataV);
            free(dataC);

            for (int k=0; k < mgpu.getFoveateLevel(); k++){
                for (int i=0; i < 3; i++){
                    free(stackDisp[k][i]);
                }free(stackDisp[k]);}
            free(stackDisp);

            return true;

        }else{
            finDisp = mgpu.match(cv_ptrL,cv_ptrR,foveated);
            ROS_INFO("non Fov Images matched! - in SRV");

            Mat ch1 = Mat::zeros(cv_ptrL->image.rows, cv_ptrL->image.cols, CV_32F);
            Mat ch2 = Mat::zeros(cv_ptrR->image.rows, cv_ptrR->image.cols, CV_32F);
            Mat ch3 = Mat::zeros(cv_ptrL->image.rows, cv_ptrL->image.cols,CV_32F);

            for(int i = 0; i < cv_ptrL->image.rows; i++)
                for(int j = 0; j < cv_ptrL->image.cols; j++)
                {
                    ch1.at<float>(i,j) = (float)(finDisp[0][i*cv_ptrL->image.cols+j]);
                    ch2.at<float>(i,j) = (float)(finDisp[1][i*cv_ptrL->image.cols+j]);
                    ch3.at<float>(i,j) = (float)(finDisp[2][i*cv_ptrL->image.cols+j]);
                }

            cv_imH->image = ch1.clone();
            cv_imV->image = ch2.clone();
            cv_imC->image = ch3.clone();

            // DEBUG ********************
            //int i = 100;
            //int j = 100;
            //ROS_INFO_STREAM(dataH[i][j] << " " << dataV[i][j]);
            //ROS_INFO_STREAM(cv_imH->image.at<float>(i,j) << " " << cv_imV->image.at<float>(i,j));
            // **************************

            cv_imH->header = cv_ptrL->header;
            cv_imV->header = cv_ptrR->header;
            cv_imC->header = cv_ptrL->header;
            cv_imH->encoding = sensor_msgs::image_encodings::TYPE_32FC1;
            cv_imV->encoding = sensor_msgs::image_encodings::TYPE_32FC1;
            cv_imC->encoding = sensor_msgs::image_encodings::TYPE_32FC1;

            rsp.dispH.image = *(cv_imH->toImageMsg());
            rsp.dispV.image = *(cv_imV->toImageMsg());
            rsp.dispC.image = *(cv_imC->toImageMsg());
            rsp.dispH.header = cv_ptrL->header;
            rsp.dispV.header = cv_ptrR->header;
            rsp.dispC.header = cv_ptrL->header;

            ROS_INFO("Disparities computed!");
            //saveImages("L.png","R.png",cv_ptrL->image, cv_ptrR->image, 1);

            //free mem
            for (int i =0; i<3;i++)
                free(finDisp[i]);
            free(finDisp);
            return true;
        }
    }

private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    typedef image_transport::SubscriberFilter ImageSubscriber;

    ImageSubscriber imL_sub_;
    ImageSubscriber imR_sub_;

    // ApproximateTime takes a queue size as its constructor argument, hence syncPolicy(5)
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> syncPolicy;
    message_filters::Synchronizer<syncPolicy> sync;
    // The Synchronizer filter is templated on a policy that determines how to synchronize the channels, ApproximateTime is one the two policies
    
    void saveImages(string str1, string str2, const Mat& imL, const Mat& imR, int reduceIm)
    {
        stringstream ss;
        ROS_INFO_STREAM("Path of the node: " << ros::package::getPath("ug_stereomatcher"));

        string out_imageL = ros::package::getPath("ug_stereomatcher") + "/" + str1;
        string out_imageR = ros::package::getPath("ug_stereomatche") + "/" + str2;
        ROS_INFO("Saving left image to: %s", out_imageL.c_str());
        ROS_INFO("Saving right image to: %s", out_imageR.c_str());

        if(reduceIm == 1)
        {
            Mat smallL, smallR;
            resize(imL, smallL, Size(imL.cols, imL.rows),
                   1.0, 1.0, cv::INTER_CUBIC);
            resize(imR, smallR, Size(imR.cols, imR.rows),
                   1.0, 1.0, cv::INTER_CUBIC);
            imwrite(out_imageL, smallL, compression_params);
            imwrite(out_imageR, smallR, compression_params);
        }
        else
        {
            imwrite(out_imageL, imL);//, compression_params);
            imwrite(out_imageR, imR);//, compression_params);
        }

        ROS_INFO("Images saved!");
    }
    
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "RH_GPU_matcher"); /* third argument is the name of the node */
    
    GPU_matcher matcher_(argc,argv);
    //GPU_matcher matcher_;
    
    //ros::spin(); /* calling spin to process events such as subscribing msg,services or actions*/

    while( ros::ok() )
    {
        ros::spin();
    }

    return EXIT_SUCCESS;
    
}

// In MatchGPULib.cpp
//1213
//355
//1672
//1035
//1209
//310
//312
//357
//1527
//870

//valgrind --tool=memcheck --leak-check=yes --log-file="valgrind.txt" --xml=yes --xml-file="valgrind.xml" --trace-children=yes ./RHGPU_matcher
