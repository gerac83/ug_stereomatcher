/*
Mozhgan <mozhgan.kabiri@gmail.com>
Fov image test
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

#include <ug_stereomatcher/foveatedstack.h>

#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "../gpu_matcher/MatchGPULib.h"

//namespace enc = sensor_msgs::image_encodings;

static const char FDISPARITY_H[] = "output_stackH"; // Fov disparity on the x axis
static const char FDISPARITY_V[] = "output_stackV"; // Fov disparity on the y axis
static const char FDISPARITY_C[] = "output_stackC"; // Fov confidence map


class saveDisp
{

public:

saveDisp():

dispH_sub_(nh_, FDISPARITY_H, 1),
dispV_sub_(nh_, FDISPARITY_V, 1),
dispC_sub_(nh_, FDISPARITY_C, 1),
sync_disp(syncPolicy_disp(1), dispH_sub_, dispV_sub_, dispC_sub_)

{

sync_disp.registerCallback(boost::bind(&saveDisp::getDisparities, this, _1, _2, _3));

ROS_INFO("Node initialised");

}

 //Functions
void saveImages(string str1, const Mat& imL, int reduceIm);
void getDisparities(const ug_stereomatcher::foveatedstackConstPtr dispX_msg, const ug_stereomatcher::foveatedstackConstPtr dispY_msg, const ug_stereomatcher::foveatedstackConstPtr dispC_msg);

private:
    //Variables

    //ROS related stuff
    ros::NodeHandle nh_;

typedef message_filters::sync_policies::ApproximateTime<ug_stereomatcher::foveatedstack, ug_stereomatcher::foveatedstack, ug_stereomatcher::foveatedstack> syncPolicy_disp;

    message_filters::Subscriber<ug_stereomatcher::foveatedstack> dispH_sub_;
    message_filters::Subscriber<ug_stereomatcher::foveatedstack> dispV_sub_;
    message_filters::Subscriber<ug_stereomatcher::foveatedstack> dispC_sub_;

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
 
void saveDisp::getDisparities(const ug_stereomatcher::foveatedstackConstPtr dispX_msg, const ug_stereomatcher::foveatedstackConstPtr dispY_msg, const ug_stereomatcher::foveatedstackConstPtr dispC_msg)
{
    // Get images
    cv_bridge::CvImagePtr cv_dispPtrH, cv_dispPtrV, cv_dispPtrC;

    ROS_INFO_STREAM("Rows: " << dispX_msg->im_height << " Cols: " << dispX_msg->im_width);
    ROS_INFO_STREAM("Rows: " << dispY_msg->im_height << " Cols: " << dispY_msg->im_width);
    ROS_INFO_STREAM("Rows: " << dispC_msg->im_height << " Cols: " << dispC_msg->im_width);

    ROS_INFO("Processing images...");
    try
    {
        cv_dispPtrH = cv_bridge::toCvCopy(dispX_msg->image_stack, enc::TYPE_32FC1);
        cv_dispPtrV = cv_bridge::toCvCopy(dispY_msg->image_stack, enc::TYPE_32FC1);
        cv_dispPtrC = cv_bridge::toCvCopy(dispC_msg->image_stack, enc::TYPE_32FC1);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert disparities to 'TYPE_32FC1'");
        return;
    }

saveImages("_FC.tif", cv_dispPtrC->image,1);
saveImages("_FH.tif", cv_dispPtrH->image,1);
saveImages("_FV.tif", cv_dispPtrV->image,1);
}

/* *************** MAIN PROGRAM *************** */
int main(int argc, char **argv)
{
    ros::init(argc, argv, "RH_FdispImgTest");

    saveDisp sD_;
    
    ros::spin();

    return EXIT_SUCCESS;

}
