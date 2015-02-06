/*
Mozhgan <mozhgan.kabiri@gmail.com>
Non Fov image test
*/
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ros/package.h>

#include <ug_stereomatcher/ug_stereomatcher.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include <stereo_msgs/DisparityImage.h>

#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "../gpu_matcher/MatchGPULib.h"

//namespace enc = sensor_msgs::image_encodings;

static const char DISPARITY_H[] = "output_disparityH"; // nFov disparity on the x axis
static const char DISPARITY_V[] = "output_disparityV"; // nFov disparity on the y axis
static const char DISPARITY_C[] = "output_disparityC"; // Confidence map

class saveDisp
{

public:

saveDisp():

dispH_sub_(nh_, DISPARITY_H, 1),
dispV_sub_(nh_, DISPARITY_V, 1),
dispC_sub_(nh_, DISPARITY_C, 1),
sync_disp(syncPolicy_disp(1), dispH_sub_, dispV_sub_, dispC_sub_)

{

sync_disp.registerCallback(boost::bind(&saveDisp::getDisparities, this, _1, _2, _3));

ROS_INFO("Node initialised");

}

 //Functions
void saveImages(string str1, const Mat& imL, int reduceIm);
void getDisparities(const stereo_msgs::DisparityImageConstPtr& dispX_msg, const stereo_msgs::DisparityImageConstPtr& dispY_msg, const stereo_msgs::DisparityImageConstPtr& dispC_msg);

private:
    //Variables

    //ROS related stuff
    ros::NodeHandle nh_;

typedef message_filters::sync_policies::ApproximateTime<stereo_msgs::DisparityImage, stereo_msgs::DisparityImage, stereo_msgs::DisparityImage> syncPolicy_disp;

    message_filters::Subscriber<stereo_msgs::DisparityImage> dispH_sub_;
    message_filters::Subscriber<stereo_msgs::DisparityImage> dispV_sub_;
    message_filters::Subscriber<stereo_msgs::DisparityImage> dispC_sub_;

    message_filters::Synchronizer<syncPolicy_disp> sync_disp;
   
};

    void saveDisp::saveImages(string str1, const Mat& im, int reduceIm)
    {
        stringstream ss;
        ROS_INFO_STREAM("Path of the node: " << ros::package::getPath("ug_stereomatcher"));

        string out_image = ros::package::getPath("ug_stereomatcher") + "/" + str1;

        ROS_INFO("Saving image to: %s", out_image.c_str());
   
        imwrite(out_image, im);//, compression_params);
 
        ROS_INFO("Images saved!");
    }
 
void saveDisp::getDisparities(const stereo_msgs::DisparityImageConstPtr& dispX_msg, const stereo_msgs::DisparityImageConstPtr& dispY_msg, const stereo_msgs::DisparityImageConstPtr& dispC_msg)
{
    // Get images
    cv_bridge::CvImagePtr cv_dispPtrH, cv_dispPtrV, cv_dispPtrC;

    ROS_INFO_STREAM("Rows: " << dispX_msg->image.height << " Cols: " << dispX_msg->image.width);
    ROS_INFO_STREAM("Rows: " << dispY_msg->image.height << " Cols: " << dispY_msg->image.width);
    ROS_INFO_STREAM("Rows: " << dispC_msg->image.height << " Cols: " << dispC_msg->image.width);

    ROS_INFO("Processing images...");
    try
    {
        cv_dispPtrH = cv_bridge::toCvCopy(dispX_msg->image, enc::TYPE_32FC1);
        cv_dispPtrV = cv_bridge::toCvCopy(dispY_msg->image, enc::TYPE_32FC1);
        cv_dispPtrC = cv_bridge::toCvCopy(dispC_msg->image, enc::TYPE_32FC1);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert disparities to 'TYPE_32FC1'");
        return;
    }

saveImages("_C.tif", cv_dispPtrC->image,1);
saveImages("_H.tif", cv_dispPtrH->image,1);
saveImages("_V.tif", cv_dispPtrV->image,1);
}

/* *************** MAIN PROGRAM *************** */
int main(int argc, char **argv)
{
    ros::init(argc, argv, "RH_dispImgTest");

    saveDisp sD_;
    
    ros::spin();

    return EXIT_SUCCESS;

}
