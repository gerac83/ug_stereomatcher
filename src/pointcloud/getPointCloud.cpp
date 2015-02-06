/*
By : Mozhgan  & Gerardo
FOVEA related notes : 

- original pyramid : 
1 - level 0 is full size image (16Mpixel) and level 6 has the fovea size. 
2 - From Level 0 to 1, 2, 3 , .. , we divide the size of each the image at each level by sqrt(2).

- fovea pyramid : 
1 - level 0 has finer detail. It is the zoomed in image. 
2 - From Level 0 to 1, 2, 3 , .. , the level 6 has coarse detail. It is the same as the original image but
lower reolustion.

*The fovea disparity array stores data for each level one after another in the shape of vertical stack image.
So, level 0 of fovea pyramid that has finer detail, is stored first in the array.

Now about the destination level and source level while mapping the x and y coordinates :

1 - The term 'LEVEL' in the fovea pyramid is called 'source level' and in the original pyramid, it is called
    'destination level'.
2 - For the fovea point cloud, we take the image at the 'srcLevel' in fovea pyramid and map the x and y to
    the image at the 'destLevel' in the original pyramid.
3 - By default, we assume the destination level is 0. i.e : So, we map x and y to the full size image (16Mega pixel).
4 - If you choose to change the destination level to any other level between 0 to 6, note that the image
    size that is already is being used (16 Megapixel), is no longer valid and the size should be the same
    as the image at the destLevel in original pyramid.
    - if the destLevel is 5 and source level is 2, then we will have a set of different calculations. E.g : we have
      to divide the level by square root 2, to the power of (5-2) and so on.
    - the default has been set to be working with either of these levels. But to map the colors, it currently works
      for the destLeve = 0 which is the 16Mega pixel image.
Last update : 22.12.14

Update 06.02.15
1 - destLevel is always initialised with 0. No need to specify the destination level anymore.

*/
#include <ug_stereomatcher/ug_stereomatcher.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include <stereo_msgs/DisparityImage.h>
#include <ug_stereomatcher/foveatedstack.h>

#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include "pcl_ros/point_cloud.h"

#include <math.h>
#include <iostream>
#include <string>
#include <sstream>

#include <vector>

using namespace std; 

#define SCALE 1.41421356
#define MAX_LEVEL 14    
#define foveaLevel 7

// Messages
static const char CAM_SUB_LEFT[] = "input_left_image";
static const char CAM_SUB_RIGHT[] = "input_right_image";
static const char CAMERA_INFO_L[] = "camera_info_left";
static const char CAMERA_INFO_R[] = "camera_info_right";
static const char DISPARITY_H[] = "output_disparityH"; // nFov disparity on the x axis
static const char DISPARITY_V[] = "output_disparityV"; // nFov disparity on the y axis
static const char FDISPARITY_H[] = "output_stackH"; // Fov disparity on the x axis
static const char FDISPARITY_V[] = "output_stackV"; // Fov disparity on the y axis
static const char POINT_CLOUD[] = "output_pointcloud";
static const char POINT_CLOUD_RES[] = "output_pointcloud_resized";

// Parameter Server
static const char FOVEATEDLev_src[] = "fovLevel";
static const char FOVEATEDQ[] = "foveated";

class CdynamicCalibration
{

public:
    //Variables
    ros::Publisher output;

    bool save_cloud;
    int sampling;
    float resizeFactor;

    int foveated;

    // P1, projection matrix from left camera, P2, projection matrix from right camera
    Mat_<double> K1_, D1_, P1_, F_;
    Mat_<double> K2_, D2_, P2_;
    Mat imgL, imgR, dispX, dispY,fdispX, fdispY;
    sensor_msgs::CameraInfo cam_infoL, cam_infoR;

    ros::Publisher pub_cloud_;
    ros::Publisher pub_cloud_resized_;

    CdynamicCalibration() :
        it_(nh_),
        imL_sub_(it_, CAM_SUB_LEFT, 5),
        imR_sub_(it_, CAM_SUB_RIGHT, 5), // TODO: include PTU and tf broadcaster
        infoL_sub_(nh_, CAMERA_INFO_L, 5),
        infoR_sub_(nh_, CAMERA_INFO_R, 5),
        dispH_sub_(nh_, DISPARITY_H, 5),
        dispV_sub_(nh_, DISPARITY_V, 5),
        fdispH_sub_(nh_, FDISPARITY_H, 5),
        fdispV_sub_(nh_, FDISPARITY_V, 5),
        sync_imgs(syncPolicy_imgs(5), imL_sub_, imR_sub_, infoL_sub_, infoR_sub_),
        sync_disp(syncPolicy_disp(5), dispH_sub_, dispV_sub_),
        sync_fdisp(syncPolicy_fdisp(5), fdispH_sub_, fdispV_sub_)
    {
        
        if (nh_.hasParam(FOVEATEDQ)){
            nh_.getParam(FOVEATEDQ, foveated);
        }
        else{
            ROS_WARN("foveated option has not been set. Program exits");
            foveated = 0; // 0 : non foveated , 1 : foveated
            //return;
        }
 
        resizeFactor = 0.2;

        pub_cloud_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB> >(POINT_CLOUD, 1);
        pub_cloud_resized_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB> >(POINT_CLOUD_RES, 1);

        sync_imgs.registerCallback(boost::bind(&CdynamicCalibration::getImages, this, _1, _2, _3, _4));

        if (foveated == 1)
            sync_fdisp.registerCallback(boost::bind(&CdynamicCalibration::getFDisparities, this, _1, _2));
        else
            sync_disp.registerCallback(boost::bind(&CdynamicCalibration::getDisparities, this, _1, _2));  

        ROS_INFO("Node initialised");
    }
    //Functions
    void getImages(const sensor_msgs::ImageConstPtr& imL, const sensor_msgs::ImageConstPtr& imR, const sensor_msgs::CameraInfoConstPtr& msgL, const sensor_msgs::CameraInfoConstPtr& msgR);

    void getDisparities(const stereo_msgs::DisparityImageConstPtr& dispX_msg, const stereo_msgs::DisparityImageConstPtr& dispY_msg); // non Fovea 
    void getFDisparities(const ug_stereomatcher::foveatedstackConstPtr dispX_msg, const ug_stereomatcher::foveatedstackConstPtr dispY_msg); // Fovea

private:
    //Variables
    int fovH; // fovea Height
    int fovW; // fovea Width

    int srcLevel; // source level in fovea pyramid
    int destLevel; // destination level in original pyramid - (by default it is Zero ; i.e : Full resolution level in original pyramid is 'level zero')

    int leftMargin_offset,upperMargin_offset;
    int heightInit,widthInit;

    //ROS related stuff
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;

    typedef image_transport::SubscriberFilter ImageSubscriber;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> syncPolicy_imgs;
    typedef message_filters::sync_policies::ApproximateTime<stereo_msgs::DisparityImage, stereo_msgs::DisparityImage> syncPolicy_disp;
    typedef message_filters::sync_policies::ApproximateTime<ug_stereomatcher::foveatedstack, ug_stereomatcher::foveatedstack> syncPolicy_fdisp;

    ImageSubscriber imL_sub_;
    ImageSubscriber imR_sub_;
    message_filters::Subscriber<sensor_msgs::CameraInfo> infoL_sub_;
    message_filters::Subscriber<sensor_msgs::CameraInfo> infoR_sub_;
    message_filters::Subscriber<stereo_msgs::DisparityImage> dispH_sub_;
    message_filters::Subscriber<stereo_msgs::DisparityImage> dispV_sub_;

    message_filters::Subscriber<ug_stereomatcher::foveatedstack> fdispH_sub_;
    message_filters::Subscriber<ug_stereomatcher::foveatedstack> fdispV_sub_;
 
    message_filters::Synchronizer<syncPolicy_imgs> sync_imgs;
    message_filters::Synchronizer<syncPolicy_disp> sync_disp;
    message_filters::Synchronizer<syncPolicy_fdisp> sync_fdisp;

    //Functions
    void printMatrix(Mat M, bool printType);
    std::string getImageType(int number);
    void saveXYZ(const char* filename, const Mat& mat);
    bool getCameraInfo(const sensor_msgs::CameraInfoConstPtr& msgL, const sensor_msgs::CameraInfoConstPtr& msgR);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr doReconstructionRGB();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr doReconstructionRGB_FOV();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr doReconstruction_resized();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr doReconstructionFOV_resized();
    vector<float> get3DPoint(int xx, int yy);
    float getRangePoint(int xx, int yy);
    float getRangePointFOV(int xx, int yy);

    void set_hInit (int heightInit);
    void set_wInit (int widthInit);
    int get_wInit();
    int get_hInit();

    /* New functions to be used instead of 'leftMargin','upperMargin','originalX','originalY'*/
    int left_marginOf_in(int srcLevel, int destLevel);
    int upper_marginOf_in(int srcLevel, int destLevel);
    float mapXcoord(int srcLevel, int destLevel,int srcX);
    float mapYcoord(int srcLevel, int destLevel,int srcY);
};

void CdynamicCalibration::set_hInit (int h){
    heightInit = h;
}
void CdynamicCalibration::set_wInit (int w){
    widthInit = w;
}
int CdynamicCalibration::get_wInit(){
    return widthInit;
}
int CdynamicCalibration::get_hInit(){
    return heightInit;
}

void CdynamicCalibration::getImages(const sensor_msgs::ImageConstPtr& imL, const sensor_msgs::ImageConstPtr& imR, const sensor_msgs::CameraInfoConstPtr& msgL, const sensor_msgs::CameraInfoConstPtr& msgR)
{
    // Get images
    cv_bridge::CvImagePtr cv_ptrL, cv_ptrR;
    ROS_INFO("Getting images and camera info...");
    try
    {
        cv_ptrL = cv_bridge::toCvCopy(imL, enc::RGB8);
        cv_ptrR = cv_bridge::toCvCopy(imR, enc::RGB8);
    }
    catch (cv_bridge::Exception& e)
    {
        //ROS_ERROR("Could not convert from '%s' to 'rgb8'.", imL->encoding.c_str());
        return;
    }

    imgL = cv_ptrL->image.clone();
    imgR = cv_ptrR->image.clone();

    //ROS_INFO_STREAM("imgL size"<<cv_ptrL->image.size()); // [4928 x 3264]

    int heightInit = (int) cv_ptrL->image.rows;
    int widthInit = (int) cv_ptrL->image.cols;

    //ROS_INFO_STREAM("Image (h(rows),w(cols)) = " << heightInit << " , " << widthInit );
    /* Setting the width and height of the original left image */
    set_hInit (heightInit);
    set_wInit (widthInit);

    ROS_INFO("Getting camera info");

    if(getCameraInfo(msgL, msgR))
    {
        ROS_INFO("Camera information read succesfully");
    }
    else
    {
        ROS_WARN("Camera information could not be read");
    }

    ROS_INFO("Done!!");

    return;

}

/* Get disparities for the non foveated full resolution */
void CdynamicCalibration::getDisparities(const stereo_msgs::DisparityImageConstPtr& dispX_msg, const stereo_msgs::DisparityImageConstPtr& dispY_msg)
{
    // Get images
    cv_bridge::CvImagePtr cv_dispPtrH, cv_dispPtrV;

    ROS_INFO_STREAM("Rows: " << dispX_msg->image.height << " Cols: " << dispX_msg->image.width);
    ROS_INFO_STREAM("Rows: " << dispY_msg->image.height << " Cols: " << dispY_msg->image.width);

    ROS_INFO("Processing images...");
    try
    {
        cv_dispPtrH = cv_bridge::toCvCopy(dispX_msg->image, enc::TYPE_32FC1);
        cv_dispPtrV = cv_bridge::toCvCopy(dispY_msg->image, enc::TYPE_32FC1);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert disparities to 'TYPE_32FC1'");
        return;
    }

    dispX = cv_dispPtrH->image.clone();
    dispY = cv_dispPtrV->image.clone();

    //ROS_INFO_STREAM("dispH size"<<cv_dispPtrH->image.size());

    cv_dispPtrH->image.release();
    cv_dispPtrV->image.release();

    ROS_INFO("Creating Point Cloud...");
    if(dispX.size() != dispY.size())
    {
        ROS_ERROR("Disparity images are not the same in dimensions");
        return;
    }

    ROS_INFO_STREAM("Dimension of nfoveated disparity images: " << dispX.rows << ", " << dispX.cols);

    if(pub_cloud_.getNumSubscribers() == 0)
    {
        ROS_WARN("No one is asking for a point cloud :)");
    }
    else
    {
        ROS_INFO("Computing point cloud!!");
        int64 t = getTickCount();

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr;
        point_cloud_ptr = doReconstructionRGB();

        t = getTickCount() - t;
        ROS_INFO_STREAM(" DONE!");
        ROS_INFO("Time elapsed: %fms\n", t*1000/getTickFrequency());

        point_cloud_ptr->header.frame_id = "left_camera";
        point_cloud_ptr->width = (int) point_cloud_ptr->points.size();
        point_cloud_ptr->height = 1;

        ROS_INFO_STREAM("Size point cloud: " << (int) point_cloud_ptr->points.size());

        if(save_cloud)
        {
            ROS_INFO("Saving point cloud to a file");
            pcl::io::savePCDFileASCII ("test_pcd.pcd", *point_cloud_ptr);
        }

        ROS_INFO("DONE!");
        pub_cloud_.publish(*point_cloud_ptr);

    }

    if(pub_cloud_resized_.getNumSubscribers() == 0)
    {
        ROS_WARN("No one is asking for a resized point cloud :)");
    }
    else
    {
        ROS_INFO("Computing resized point cloud!!");
        int64 t = getTickCount();

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr;
        point_cloud_ptr = doReconstruction_resized();

        t = getTickCount() - t;
        ROS_INFO_STREAM(" DONE!");
        ROS_INFO("Time elapsed: %fms\n", t*1000/getTickFrequency());

        point_cloud_ptr->header.frame_id = "left_camera";
        //point_cloud_ptr->header.stamp = dispX_msg->header.stamp;
        point_cloud_ptr->width = (int) point_cloud_ptr->points.size();
        point_cloud_ptr->height = 1;

        ROS_INFO_STREAM("Size point cloud: " << (int) point_cloud_ptr->points.size());

        if(save_cloud)
        {
            ROS_INFO("Saving point cloud to a file");
            pcl::io::savePCDFileASCII ("test_pcd_resized.pcd", *point_cloud_ptr);
        }

        ROS_INFO("DONE!");
        pub_cloud_resized_.publish(*point_cloud_ptr);

    }

    ROS_INFO_STREAM("Full point cloud: " << pub_cloud_.getNumSubscribers());
    ROS_INFO_STREAM("Resized point cloud: " << pub_cloud_resized_.getNumSubscribers());

    return;

}

#define M_SQRT2   1.41421356237309504880 /* sqrt(2) */ 
#define M_SQRT1_2   0.70710678118654752440 /* 1/sqrt(2) */

/*
Mozhgan Updated : 16.12.14
 The function retrieves the 'x' coordinate in the original pyramid at 'destLevel'
 of the coordinate 'srcX' in the foveated pyramid at level 'srcLevel'.
*/
float CdynamicCalibration::mapXcoord(int srcLevel, int destLevel,int srcX){
    float sqrt_root;

    //ROS_WARN_STREAM("srcLevel: " << srcLevel << " destLevel: " << destLevel);

    if (srcLevel < destLevel)
        sqrt_root = M_SQRT1_2;
    else
        sqrt_root = M_SQRT2;

    float original_x = (float)leftMargin_offset + (float)srcX*pow(sqrt_root,(float)abs(srcLevel-destLevel));

    return original_x;
}

/*
Mozhgan Updated : 16.12.14
 The function retrieves the 'y' coordinate in the original pyramid at 'destLevel'
 of the coordinate 'srcY' in the foveated pyramid at level 'srcLevel'.
*/
float CdynamicCalibration::mapYcoord(int srcLevel, int destLevel,int srcY){

    float sqrt_root;

    //ROS_WARN_STREAM("srcLevel: " << srcLevel << " destLevel: " << destLevel);

    if (srcLevel < destLevel)
        sqrt_root = M_SQRT1_2;
    else
        sqrt_root = M_SQRT2;

    float original_y = (float)upperMargin_offset + (float)srcY*pow(sqrt_root,(float)abs(srcLevel-destLevel));

    return original_y;
}

/*
Mozhgan Updated : 16.12.14
The function retrieves the upper margin for the scaled fovea image at srcLevel, with respect to destLevel.
We need to divide the height of the image at destLevel (in original pyramid), and deduct it from the 1/2 height
of the scaled fovea image of srcLevel.

If the source level NUMBER is smaller than the destination level, then
*/
int CdynamicCalibration::upper_marginOf_in(int srcLevel, int destLevel){
    int height[MAX_LEVEL-1];
    int  rows, y_src,y1,u_src , u;

    int scaled_fovea_level = 6 - srcLevel; // default ; assuming srcLevel is always greater than destLevel

    if (srcLevel < destLevel)
        scaled_fovea_level = srcLevel + destLevel; // this will be more than 7, upto 14

    height[0] = get_hInit();

    for(int i=0; i< MAX_LEVEL; i++)
        height[i+1]=height[i]/SCALE;

    rows=height[destLevel];
    y1=rows/2;

    y_src=height[scaled_fovea_level];

    u_src=y1-y_src/2;

    ROS_INFO_STREAM( "upper = " << u_src);
    return u_src;
}
/*
Mozhgan Updated : 16.12.14
The function retrieves the lower margin for the scaled fovea image at srcLevel, with respect to destLevel.
We need to divide the width of the image at destLevel (in original pyramid), and deduct it from the 1/2 width
of the scaled fovea image of srcLevel.
*/
int CdynamicCalibration::left_marginOf_in(int srcLevel, int destLevel){
    int width[MAX_LEVEL-1];
    int cols,x_src,x1,l_src,l;
    
    int scaled_fovea_level = 6 - srcLevel; // default ; assuming srcLevel is always greater than destLevel

    if (srcLevel < destLevel)
        scaled_fovea_level = srcLevel + destLevel; // this will be more than 7, upto 14

    width[0] = get_wInit();

    for(int i=0; i< MAX_LEVEL; i++)
        width[i+1]=width[i]/SCALE;

    cols=width[destLevel];
    x1=cols/2;

    x_src=width[scaled_fovea_level];
    l_src=x1 - x_src/2;

    ROS_INFO_STREAM( "left = " << l_src);

    return l_src;
}

/* Get disparities for the foveated version */
void CdynamicCalibration::getFDisparities(const ug_stereomatcher::foveatedstackConstPtr dispX_msg, const ug_stereomatcher::foveatedstackConstPtr dispY_msg)
{
    sampling = 1;
    // Get images
    cv_bridge::CvImagePtr cv_dispPtrH, cv_dispPtrV;

    ROS_INFO_STREAM("Original image Rows: " << dispX_msg->im_height << " Original image Cols: " << dispX_msg->im_width << " Fovea image width: " << dispX_msg->roi_width << " Fovea image height: " << dispX_msg->roi_height << " levels: " << dispY_msg->num_levels);
  
    //ROS_INFO("Processing images...");
    try
    {
        cv_dispPtrH = cv_bridge::toCvCopy(dispX_msg->image_stack, enc::TYPE_32FC1);
        cv_dispPtrV = cv_bridge::toCvCopy(dispY_msg->image_stack, enc::TYPE_32FC1);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert disparities to 'TYPE_32FC1'");
        return;
    }

    /* Reading in the destination level parameter */
    if (nh_.hasParam(FOVEATEDLev_src)){
        nh_.getParam(FOVEATEDLev_src, srcLevel);
        if (srcLevel > 6 || srcLevel <0){
            ROS_WARN("Source level in the fovea pyramid is not valid (should be between 0-6). It will be set to 0 (fine detail) as a default!");
            srcLevel = 0;
        }
    }
    else{
        ROS_WARN("Source level in the fovea pyramid has not been set. It will be set to 0 as a default!");
        srcLevel = 0;
    }

    destLevel = 0;
    
    fdispX = cv_dispPtrH->image.clone();
    fdispY = cv_dispPtrV->image.clone();

    ROS_INFO_STREAM("fdispH size"<<cv_dispPtrH->image.size()); // [615 x 2849]

    cv_dispPtrH->image.release();
    cv_dispPtrV->image.release();

    ROS_INFO("Creating Point Cloud ...");
    if(fdispX.size() != fdispY.size())
    {
        ROS_ERROR("Disparity images are not the same in dimensions");
        return;
    }

    ROS_INFO_STREAM("Dimension of disparity foveated stack : " << fdispX.rows << ", " << fdispX.cols); // prints out 2849, 615 for 16Mega pixel image


    if(pub_cloud_.getNumSubscribers() == 0)
    {
        ROS_WARN("No one is asking for a point cloud :)");
    }
    else
    {

        ROS_INFO("Computing range and point cloud!!");
        int64 t = getTickCount();

        fovH = fdispX.rows / foveaLevel;
        fovW = fdispX.cols ;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr;
        point_cloud_ptr = doReconstructionRGB_FOV();

        t = getTickCount() - t;
        ROS_INFO_STREAM(" DONE!");
        ROS_INFO("Time elapsed: %fms\n", t*1000/getTickFrequency());

        point_cloud_ptr->header.frame_id = "left_camera";
        point_cloud_ptr->width = (int) point_cloud_ptr->points.size();
        point_cloud_ptr->height = 1;

        ROS_INFO_STREAM("Size point cloud (fovea level " << srcLevel <<"): " << (int) point_cloud_ptr->points.size());

        if(save_cloud)
        {
            ROS_INFO("Saving point cloud to a file");
            pcl::io::savePCDFileASCII ("test_pcd_Fov.pcd", *point_cloud_ptr);
        }

        ROS_INFO("DONE!");
        pub_cloud_.publish(*point_cloud_ptr);

    }

    if(pub_cloud_resized_.getNumSubscribers() == 0)
    {
        ROS_WARN("No one is asking for a resized point cloud :)");
    }
    else
    {
        ROS_INFO("Computing resized point cloud!!");
        int64 t = getTickCount();

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr;
        point_cloud_ptr = doReconstructionFOV_resized();

        t = getTickCount() - t;
        ROS_INFO_STREAM(" DONE!");
        ROS_INFO("Time elapsed: %fms\n", t*1000/getTickFrequency());

        point_cloud_ptr->header.frame_id = "left_camera";
        point_cloud_ptr->width = (int) point_cloud_ptr->points.size();
        point_cloud_ptr->height = 1;

        ROS_INFO_STREAM("Size point cloud (fovea level " << srcLevel <<"): " << (int) point_cloud_ptr->points.size());

        if(save_cloud)
        {
            ROS_INFO("Saving point cloud to a file");
            pcl::io::savePCDFileASCII ("test_pcd_resized_Fov.pcd", *point_cloud_ptr);
        }

        ROS_INFO("DONE!");
        pub_cloud_resized_.publish(*point_cloud_ptr);

    }
    return;
}

/* *************** PRIVATE FUNCTIONS *************** */

// Fovea construction
pcl::PointCloud<pcl::PointXYZRGB>::Ptr CdynamicCalibration::doReconstructionRGB_FOV()
{
    unsigned char pr, pg, pb;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr_ (new pcl::PointCloud<pcl::PointXYZRGB>);

    int barWidth = 70;

    leftMargin_offset = left_marginOf_in(srcLevel, destLevel);
    upperMargin_offset = upper_marginOf_in(srcLevel, destLevel);
    fovH = fdispX.rows / foveaLevel;
    fovW = fdispX.cols;

    ROS_INFO_STREAM("(leftMargin,upperMargin): " << leftMargin_offset << ", " << upperMargin_offset );

    for(int ii = 0; ii < fdispX.cols; ii++) // 615 for 16Mega pixel image
    {
        float progress = ((float)ii/(float)fdispX.cols);
        int pos = barWidth * progress;
        std::cout << "[";
        for (int i = 0; i < barWidth; ++i)
        {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r";
        std::cout.flush();

        for (int jj = 0; jj < fovH ; jj++) //fovH is 407 for 16Mega pixel image
            //        for(int jj = fovH * srcLevel; jj < fovH * (srcLevel+1) ; jj++)
        {
            if(ii % sampling == 0 && jj % sampling == 0)
            {
                vector<float> point3D = get3DPoint(ii, jj);

                //Get RGB info
                int jj_1 = mapYcoord(srcLevel,destLevel,jj);
                int ii_1 = mapXcoord(srcLevel,destLevel,ii);

                unsigned char* rgb_ptr = imgL.ptr<unsigned char>(jj_1);
                pb = rgb_ptr[3*ii_1];
                pg = rgb_ptr[3*ii_1+1];
                pr = rgb_ptr[3*ii_1+2];

                pcl::PointXYZRGB point_small;
                uint32_t rgb = (static_cast<uint32_t>(pr) << 16 | static_cast<uint32_t>(pg) << 8 | static_cast<uint32_t>(pb));

                point_small.x = point3D.at(0);
                point_small.y = point3D.at(1);
                point_small.z = point3D.at(2);

                point_small.rgb = *reinterpret_cast<float*>(&rgb);
                point_cloud_ptr_->points.push_back(point_small);
            }

        }
    }
    std::cout << std::endl;
    return point_cloud_ptr_;
}
pcl::PointCloud<pcl::PointXYZRGB>::Ptr CdynamicCalibration::doReconstructionRGB()
{
    unsigned char pr, pg, pb;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr_ (new pcl::PointCloud<pcl::PointXYZRGB>);

    int barWidth = 70;

    for(int ii = 0; ii < dispX.cols; ii++)
    {
        float progress = ((float)ii/(float)dispX.cols);
        int pos = barWidth * progress;
        std::cout << "[";
        for (int i = 0; i < barWidth; ++i)
        {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r";
        std::cout.flush();

        for(int jj = 0; jj < dispX.rows; jj++)
        {
            if(ii % sampling == 0 && jj % sampling == 0)
            {
                vector<float> point3D = get3DPoint(ii, jj);

                //Get RGB info
                unsigned char* rgb_ptr = imgL.ptr<unsigned char>(jj);// .ptr<uchar>(ii);
                pb = rgb_ptr[3*ii];
                pg = rgb_ptr[3*ii+1];
                pr = rgb_ptr[3*ii+2];

                pcl::PointXYZRGB point_small;
                uint32_t rgb = (static_cast<uint32_t>(pr) << 16 | static_cast<uint32_t>(pg) << 8 | static_cast<uint32_t>(pb));

                point_small.x = point3D.at(0);
                point_small.y = point3D.at(1);
                point_small.z = point3D.at(2);
                point_small.rgb = *reinterpret_cast<float*>(&rgb);
                point_cloud_ptr_->points.push_back(point_small);
            }

        }
    }
    std::cout << std::endl;
    return point_cloud_ptr_;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr CdynamicCalibration::doReconstruction_resized()
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr_ (new pcl::PointCloud<pcl::PointXYZRGB>);

    unsigned char pr, pg, pb;
    int barWidth = 70;
    Mat range_map(dispX.size(), CV_32F);

    ROS_INFO("Computing range map!");
    for(int ii = 0; ii < dispX.cols; ii++)
    {
        float progress = ((float)ii/(float)dispX.cols);
        int pos = barWidth * progress;
        std::cout << "[";
        for (int i = 0; i < barWidth; ++i)
        {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r";
        std::cout.flush();

        for(int jj = 0; jj < dispX.rows; jj++)
        {
            range_map.at<float>(jj,ii) = getRangePoint(ii, jj);
        }
    }
    std::cout << std::endl;
    ROS_INFO("Done!");

    ROS_INFO("Computing resized point cloud!");

    Mat res_range_map;
    resize(range_map, res_range_map, Size(dispX.cols*resizeFactor, dispX.rows*resizeFactor), 0, 0, cv::INTER_CUBIC);

    for(int ii = 0; ii < res_range_map.cols; ii++)
    {
        float progress = ((float)ii/(float)res_range_map.cols);
        int pos = barWidth * progress;
        std::cout << "[";
        for (int i = 0; i < barWidth; ++i)
        {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r";
        std::cout.flush();

        for(int jj = 0; jj < res_range_map.rows; jj++)
        {
            int xx = (int)((float)ii / resizeFactor);
            int yy = (int)((float)jj / resizeFactor);
            vector<float> point3D = get3DPoint(xx, yy);
            //Get RGB info
            unsigned char* rgb_ptr = imgL.ptr<unsigned char>(yy);// .ptr<uchar>(ii);
            pb = rgb_ptr[3*xx];
            pg = rgb_ptr[3*xx+1];
            pr = rgb_ptr[3*xx+2];

            pcl::PointXYZRGB point_small;
            uint32_t rgb = (static_cast<uint32_t>(pr) << 16 | static_cast<uint32_t>(pg) << 8 | static_cast<uint32_t>(pb));

            point_small.x = point3D.at(0);
            point_small.y = point3D.at(1);
            point_small.z = res_range_map.at<float>(jj,ii);
            point_small.rgb = *reinterpret_cast<float*>(&rgb);
            point_cloud_ptr_->points.push_back(point_small);
        }

    }
    ROS_INFO("Done!");
    std::cout << std::endl;
    return point_cloud_ptr_;

}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr CdynamicCalibration::doReconstructionFOV_resized()
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr_ (new pcl::PointCloud<pcl::PointXYZRGB>);

    unsigned char pr, pg, pb;
    int barWidth = 70;

    leftMargin_offset = left_marginOf_in(srcLevel, destLevel);
    upperMargin_offset = upper_marginOf_in(srcLevel, destLevel);
    fovH = fdispX.rows / foveaLevel;
    fovW = fdispX.cols;
    Mat range_map(Size(fovW, fovH), CV_32F);

    ROS_INFO_STREAM("(leftMargin,upperMargin): " << leftMargin_offset << ", " << upperMargin_offset );

    for(int ii = 0; ii < fdispX.cols; ii++) // 615 for 16Mega pixel image
    {
        float progress = ((float)ii/(float)fdispX.cols);
        int pos = barWidth * progress;
        std::cout << "[";
        for (int i = 0; i < barWidth; ++i)
        {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r";
        std::cout.flush();

        for (int jj = 0; jj < fovH ; jj++) // 418
            range_map.at<float>(jj,ii) = getRangePointFOV(ii, jj);

    }
    std::cout << std::endl;
    ROS_INFO("Done!");

    ROS_INFO("Computing resized point cloud!");

    Mat res_range_map;
    resize(range_map, res_range_map, Size(fovW*resizeFactor, fovH*resizeFactor), 0, 0, cv::INTER_CUBIC);

    //////////
    for(int ii = 0; ii < res_range_map.cols; ii++)
    {
        float progress = ((float)ii/(float)res_range_map.cols);
        int pos = barWidth * progress;
        std::cout << "[";
        for (int i = 0; i < barWidth; ++i)
        {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r";
        std::cout.flush();

        for(int jj = 0; jj < res_range_map.rows; jj++)
        {
            int xx = (int)((float)ii / resizeFactor);
            int yy = (int)((float)jj / resizeFactor);
            vector<float> point3D = get3DPoint(xx, yy);
            //Get RGB info
            unsigned char* rgb_ptr = imgL.ptr<unsigned char>(yy);// .ptr<uchar>(ii);
            pb = rgb_ptr[3*xx];
            pg = rgb_ptr[3*xx+1];
            pr = rgb_ptr[3*xx+2];

            pcl::PointXYZRGB point_small;
            uint32_t rgb = (static_cast<uint32_t>(pr) << 16 | static_cast<uint32_t>(pg) << 8 | static_cast<uint32_t>(pb));

            point_small.x = point3D.at(0);
            point_small.y = point3D.at(1);
            point_small.z = res_range_map.at<float>(jj,ii);
            point_small.rgb = *reinterpret_cast<float*>(&rgb);
            point_cloud_ptr_->points.push_back(point_small);
        }

    }
    ROS_INFO("Done!");
    std::cout << std::endl;
    return point_cloud_ptr_;

}

vector<float> CdynamicCalibration::get3DPoint(int xx, int yy)
{
    vector<float> out;
    out.resize(0);
    float x1,x2,y1,y2;

    if(foveated ==1){

        float corsp_yy = mapYcoord(srcLevel,destLevel,yy);
        float corsp_xx = mapXcoord(srcLevel,destLevel,xx);

        x1 = corsp_xx; // X1left
        y1 = corsp_yy; // X2left

        //x2 = corsp_xx + fdispX.at<float>(yy + fovH * srcLevel,xx);
        //y2 = corsp_yy + fdispY.at<float>(yy + fovH * srcLevel,xx);

        x2 = mapXcoord(srcLevel,destLevel, xx + fdispX.at<float>(yy + fovH * srcLevel,xx));
        y2 = mapYcoord(srcLevel,destLevel, yy + fdispY.at<float>(yy + fovH * srcLevel,xx));

        // x2 = corsp_xx + fdispX.at<float>(yy,xx);
        // y2 = corsp_yy + fdispY.at<float>(yy,xx);
    }
    else {
        x1 = xx; // X1left
        y1 = yy; // X2left
        x2 = xx + dispX.at<float>(yy,xx);
        y2 = yy + dispY.at<float>(yy,xx);
    }

    float a, b, c, d, e, f, g, h, i, j, x, y;
    a = (float)P1_.at<double>(0,0);                             //p11
    b = (float)(P1_.at<double>(0,2) - x1);                      //p13-X1left
    c = (float)P1_.at<double>(1,1);                             //p22
    d = (float)(P1_.at<double>(1,2) - y1);                      //p23-X2left
    e = (float)(P2_.at<double>(0,0) - x2*P2_.at<double>(2,0));
    f = (float)(P2_.at<double>(0,1) - x2*P2_.at<double>(2,1));
    g = (float)(P2_.at<double>(0,2) - x2*P2_.at<double>(2,2));
    h = (float)(P2_.at<double>(1,0) - y2*P2_.at<double>(2,0));
    i = (float)(P2_.at<double>(1,1) - y2*P2_.at<double>(2,1));
    j = (float)(P2_.at<double>(1,2) - y2*P2_.at<double>(2,2));
    x = (float)(x2*P2_.at<double>(2,3) - P2_.at<double>(0,3));
    y = (float)(y2*P2_.at<double>(2,3) - P2_.at<double>(1,3));

    float XUp = (d*f*h - c*g*h - d*e*i + c*e*j)*(-(d*i*x) + c*j*x + d*f*y - c*g*y) +
                pow(b,2.0)*((f*h - e*i)*(-(i*x) + f*y) + pow(c,2.0)*(e*x + h*y)) +
                a*b*((-(g*i) + f*j)*(i*x - f*y) + c*d*(f*x + i*y) - pow(c,2.0)*(g*x + j*y));
    float YUp = (pow(b,2.0)*(f*h - e*i) + d*(d*f*h - c*g*h - d*e*i + c*e*j))*(h*x - e*y) +
                a*b*((c*d*e + g*h*i - 2.0*f*h*j + e*i*j)*x + (c*d*h + f*g*h - 2.0*e*g*i + e*f*j)*y) +
                pow(a,2.0)*((g*i - f*j)*(-(j*x) + g*y) + pow(d,2.0)*(f*x + i*y) - c*d*(g*x + j*y));
    float ZUp = c*(-(d*f*h) + c*g*h + d*e*i - c*e*j)*(h*x - e*y) -a*b*((f*h - e*i)*(-(i*x) + f*y) +
                pow(c,2.0)*(e*x + h*y)) + pow(a,2.0)*((g*i - f*j)*(i*x - f*y) - c*d*(f*x + i*y) +
                pow(c,2.0)*(g*x + j*y));
    float divisor = pow(b,2.0)*(pow(c,2.0)*(pow(e,2.0) + pow(h,2.0)) + pow(f*h - e*i,2.0)) +
                    pow(d*f*h - c*g*h - d*e*i + c*e*j,2.0) - 2.0*a*b*(-(c*d*(e*f + h*i)) +
                    (f*h - e*i)*(-(g*i) + f*j) + pow(c,2.0)*(e*g + h*j)) + pow(a,2.0)*
                    (pow(d,2.0)*(pow(f,2.0) + pow(i,2.0)) + pow(g*i - f*j,2.0) - 2.0*c*d*(f*g + i*j) +
                    pow(c,2.0)*(pow(g,2.0) + pow(j,2.0)));

    out.push_back(XUp/divisor);
    out.push_back(YUp/divisor);
    out.push_back(ZUp/divisor);
    return out;
}

float CdynamicCalibration::getRangePoint(int xx, int yy)
{
    float x1 = xx;
    float y1 = yy;
    float x2 = xx + dispX.at<float>(yy,xx);
    float y2 = yy + dispY.at<float>(yy,xx);

    float a, b, c, d, e, f, g, h, i, j, x, y;
    a = (float)P1_.at<double>(0,0);
    b = (float)(P1_.at<double>(0,2) - x1);
    c = (float)P1_.at<double>(1,1);
    d = (float)(P1_.at<double>(1,2) - y1);
    e = (float)(P2_.at<double>(0,0) - x2*P2_.at<double>(2,0));
    f = (float)(P2_.at<double>(0,1) - x2*P2_.at<double>(2,1));
    g = (float)(P2_.at<double>(0,2) - x2*P2_.at<double>(2,2));
    h = (float)(P2_.at<double>(1,0) - y2*P2_.at<double>(2,0));
    i = (float)(P2_.at<double>(1,1) - y2*P2_.at<double>(2,1));
    j = (float)(P2_.at<double>(1,2) - y2*P2_.at<double>(2,2));
    x = (float)(x2*P2_.at<double>(2,3) - P2_.at<double>(0,3));
    y = (float)(y2*P2_.at<double>(2,3) - P2_.at<double>(1,3));

    float ZUp = c*(-(d*f*h) + c*g*h + d*e*i - c*e*j)*(h*x - e*y) -a*b*((f*h - e*i)*(-(i*x) + f*y) +
                pow(c,2.0)*(e*x + h*y)) + pow(a,2.0)*((g*i - f*j)*(i*x - f*y) - c*d*(f*x + i*y) +
                pow(c,2.0)*(g*x + j*y));
    float divisor = pow(b,2.0)*(pow(c,2.0)*(pow(e,2.0) + pow(h,2.0)) + pow(f*h - e*i,2.0)) +
                    pow(d*f*h - c*g*h - d*e*i + c*e*j,2.0) - 2.0*a*b*(-(c*d*(e*f + h*i)) +
                    (f*h - e*i)*(-(g*i) + f*j) + pow(c,2.0)*(e*g + h*j)) + pow(a,2.0)*
                    (pow(d,2.0)*(pow(f,2.0) + pow(i,2.0)) + pow(g*i - f*j,2.0) - 2.0*c*d*(f*g + i*j) +
                    pow(c,2.0)*(pow(g,2.0) + pow(j,2.0)));

    return ZUp/divisor;
}

float CdynamicCalibration::getRangePointFOV(int xx, int yy)
{
    float x1,x2,y1,y2;

    float corsp_yy = mapYcoord(srcLevel,destLevel,yy);
    float corsp_xx = mapXcoord(srcLevel,destLevel,xx);

    x1 = corsp_xx; // X1left
    y1 = corsp_yy; // X2left

    x2 = mapXcoord(srcLevel,destLevel, xx + fdispX.at<float>(yy + fovH * srcLevel,xx));
    y2 = mapYcoord(srcLevel,destLevel, yy + fdispY.at<float>(yy + fovH * srcLevel,xx));

    float a, b, c, d, e, f, g, h, i, j, x, y;
    a = (float)P1_.at<double>(0,0);
    b = (float)(P1_.at<double>(0,2) - x1);
    c = (float)P1_.at<double>(1,1);
    d = (float)(P1_.at<double>(1,2) - y1);
    e = (float)(P2_.at<double>(0,0) - x2*P2_.at<double>(2,0));
    f = (float)(P2_.at<double>(0,1) - x2*P2_.at<double>(2,1));
    g = (float)(P2_.at<double>(0,2) - x2*P2_.at<double>(2,2));
    h = (float)(P2_.at<double>(1,0) - y2*P2_.at<double>(2,0));
    i = (float)(P2_.at<double>(1,1) - y2*P2_.at<double>(2,1));
    j = (float)(P2_.at<double>(1,2) - y2*P2_.at<double>(2,2));
    x = (float)(x2*P2_.at<double>(2,3) - P2_.at<double>(0,3));
    y = (float)(y2*P2_.at<double>(2,3) - P2_.at<double>(1,3));

    float ZUp = c*(-(d*f*h) + c*g*h + d*e*i - c*e*j)*(h*x - e*y) -a*b*((f*h - e*i)*(-(i*x) + f*y) +
                pow(c,2.0)*(e*x + h*y)) + pow(a,2.0)*((g*i - f*j)*(i*x - f*y) - c*d*(f*x + i*y) +
                pow(c,2.0)*(g*x + j*y));
    float divisor = pow(b,2.0)*(pow(c,2.0)*(pow(e,2.0) + pow(h,2.0)) + pow(f*h - e*i,2.0)) +
                    pow(d*f*h - c*g*h - d*e*i + c*e*j,2.0) - 2.0*a*b*(-(c*d*(e*f + h*i)) +
                    (f*h - e*i)*(-(g*i) + f*j) + pow(c,2.0)*(e*g + h*j)) + pow(a,2.0)*
                    (pow(d,2.0)*(pow(f,2.0) + pow(i,2.0)) + pow(g*i - f*j,2.0) - 2.0*c*d*(f*g + i*j) +
                    pow(c,2.0)*(pow(g,2.0) + pow(j,2.0)));

    return ZUp/divisor;
}

void CdynamicCalibration::printMatrix(Mat M, bool printType)
{
    if(printType)
        ROS_INFO_STREAM("Matrix type: " << getImageType(M.type()));
    // dont print empty matrices
    if (M.empty()){
        ROS_INFO("---");
        return;
    }
    // loop through columns and rows of the matrix
    for(int i=0; i < M.rows; i++){
        for(int j=0; j < M.cols ; j++){
            if(M.type() == 6)
                cout << M.at<double>(i,j) << "\t";
            else
                cout << M.at<float>(i,j) << "\t";
        }
        cout<<endl;
    }
    cout<<endl;
}

std::string CdynamicCalibration::getImageType(int number)
{
    // find type
    int imgTypeInt = number%8;
    std::string imgTypeString;

    switch (imgTypeInt)
    {
    case 0:
        imgTypeString = "8U";
        break;
    case 1:
        imgTypeString = "8S";
        break;
    case 2:
        imgTypeString = "16U";
        break;
    case 3:
        imgTypeString = "16S";
        break;
    case 4:
        imgTypeString = "32S";
        break;
    case 5:
        imgTypeString = "32F";
        break;
    case 6:
        imgTypeString = "64F";
        break;
    default:
        break;
    }

    // find channel
    int channel = (number/8) + 1;

    std::stringstream type;
    type<<"CV_"<<imgTypeString<<"C"<<channel;

    return type.str();
}

void CdynamicCalibration::saveXYZ(const char* filename, const Mat& mat)
{
    const float max_z = 1.0e4;
    FILE* fp = fopen(filename, "wt");
    for(int y = 0; y < mat.rows; y++)
    {
        for(int x = 0; x < mat.cols; x++)
        {
            Vec3f point = mat.at<Vec3f>(y, x);
            if(fabs(point[2] - max_z) < DBL_EPSILON || fabs(point[2]) > max_z)
                continue;

            if(point[2] < 0)
                continue;

            fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
        }
    }
    fclose(fp);
}


bool CdynamicCalibration::getCameraInfo(const sensor_msgs::CameraInfoConstPtr& msgL, const sensor_msgs::CameraInfoConstPtr& msgR)
{
    try
    {
        // Update time stamp (and frame_id if that changes for some reason)

        cam_infoL.height = msgL->height;
        cam_infoL.width = msgL->width;
        cam_infoL.D = msgL->D;
        cam_infoL.K = msgL->K;
        cam_infoL.P = msgL->P;
        //cam_infoL.F = msgL->F;

        cam_infoR.height = msgR->height;
        cam_infoR.width = msgR->width;
        cam_infoR.D = msgR->D;
        cam_infoR.K = msgR->K;
        cam_infoR.P = msgR->P;
        //cam_infoR.F = msgR->F;

        Mat K1, D1, P1, K2, D2, P2;//, F;

        D1 = Mat::zeros(1,5,CV_64F);
        K1 = Mat::zeros(3,3,CV_64F);
        P1 = Mat::zeros(3,4,CV_64F);

        D2 = Mat::zeros(1,5,CV_64F);
        K2 = Mat::zeros(3,3,CV_64F);
        P2 = Mat::zeros(3,4,CV_64F);

        //F = Mat::zeros(3,3,CV_64F);

        for(int i = 0; i < D1.cols; i++)
        {
            D1.at<double>(0,i) = msgL->D[i];
            D2.at<double>(0,i) = msgR->D[i];

            if(i < K1.rows)
            {
                for(int j = 0 ; j < P1.cols; j++)
                {
                    if(j < K1.cols)
                    {
                        K1.at<double>(i,j) = msgL->K[3*i+j];
                        K2.at<double>(i,j) = msgR->K[3*i+j];
                    }
                    P1.at<double>(i,j) = msgL->P[4*i+j];
                    P2.at<double>(i,j) = msgR->P[4*i+j];
                }
            }
        }

        D1_ = D1.clone();
        K1_ = K1.clone();
        P1_ = P1.clone();

        D2_ = D2.clone();
        K2_ = K2.clone();
        P2_ = P2.clone();

    }
    catch (...)
    {
        return false;
    }

    return true;

}

/* *************** MAIN PROGRAM *************** */
int main(int argc, char* argv[])
{
    ros::init( argc, argv, "RH_dynamicCalibration_node" );
    CdynamicCalibration rU_;

    rU_.sampling =10; // for the non fovea version change this to 5

    if((argc == 2) && (strcmp(argv[1], "-s") == 0))
    {
        ROS_INFO("Saving point cloud option is enabled!");
        rU_.save_cloud = true;
    }
    else
    {
        ROS_INFO("Use -s to save point cloud to disk");
        ROS_INFO("Saving point cloud option is not enabled!");

        // Debug Moz : enabling saving the point cloud
        rU_.save_cloud = false;
        /**/
    }

    ROS_INFO_STREAM("Sampling every: " << rU_.sampling << " pixels!");

    ros::MultiThreadedSpinner s(2);
    ros::spin(s);

    return EXIT_SUCCESS;
}

// SLOW IMPLEMENTATION OF STEREO TRIANGULATION
//            Mat M(4,4,CV_32F);
//            vector<float> p1, p2;

//            p1.push_back(ii); // x1
//            p1.push_back(ji); // y1
//            p2.push_back(ii + dispX.at<float>(jj,ii)); // x2
//            p2.push_back(jj + dispY.at<float>(jj,ii)); // y2

//            M.at<float>(0,0) = (float)P1_.at<double>(0,0) - (p1.at(0) * (float)P1_.at<double>(2,0));
//            M.at<float>(0,1) = (float)P1_.at<double>(0,1) - (p1.at(0) * (float)P1_.at<double>(2,1));
//            M.at<float>(0,2) = (float)P1_.at<double>(0,2) - (p1.at(0) * (float)P1_.at<double>(2,2));
//            M.at<float>(0,3) = (float)P1_.at<double>(0,3) - (p1.at(0) * (float)P1_.at<double>(2,3));

//            M.at<float>(1,0) = (float)P1_.at<double>(1,0) - (p1.at(1) * (float)P1_.at<double>(2,0));
//            M.at<float>(1,1) = (float)P1_.at<double>(1,1) - (p1.at(1) * (float)P1_.at<double>(2,1));
//            M.at<float>(1,2) = (float)P1_.at<double>(1,2) - (p1.at(1) * (float)P1_.at<double>(2,2));
//            M.at<float>(1,3) = (float)P1_.at<double>(1,3) - (p1.at(1) * (float)P1_.at<double>(2,3));

//            M.at<float>(2,0) = (float)P2_.at<double>(0,0) - (p2.at(0) * (float)P2_.at<double>(2,0));
//            M.at<float>(2,1) = (float)P2_.at<double>(0,1) - (p2.at(0) * (float)P2_.at<double>(2,1));
//            M.at<float>(2,2) = (float)P2_.at<double>(0,2) - (p2.at(0) * (float)P2_.at<double>(2,2));
//            M.at<float>(2,3) = (float)P2_.at<double>(0,3) - (p2.at(0) * (float)P2_.at<double>(2,3));

//            M.at<float>(3,0) = (float)P2_.at<double>(1,0) - (p2.at(1) * (float)P2_.at<double>(2,0));
//            M.at<float>(3,1) = (float)P2_.at<double>(1,1) - (p2.at(1) * (float)P2_.at<double>(2,1));
//            M.at<float>(3,2) = (float)P2_.at<double>(1,2) - (p2.at(1) * (float)P2_.at<double>(2,2));
//            M.at<float>(3,3) = (float)P2_.at<double>(1,3) - (p2.at(1) * (float)P2_.at<double>(2,3));

//            SVD svd(M);

//            float div = svd.vt.at<float>(3,3);
//            px = svd.vt.at<float>(3,0)/div;
//            py = svd.vt.at<float>(3,1)/div;
//            pz = svd.vt.at<float>(3,2)/div;
